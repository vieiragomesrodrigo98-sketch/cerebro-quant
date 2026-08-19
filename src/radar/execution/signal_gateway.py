"""
Camada de execução — cliente do relay de sinais ("Signal Connection" / MT5).

O que isto é, e o que NÃO é
---------------------------
Este módulo fala HTTP com um serviço que o PROVEDOR já opera (`POST
/v1/signals`), que roteia o sinal para as estratégias ativas configuradas do
lado da plataforma dele. **Não existe aqui, nem em lugar nenhum do projeto, a
lib `MetaTrader5`** — a ponte Windows/MT5 já está resolvida do lado deles;
o nosso lado só precisa falar o contrato HTTP.

⚠️ Isto fala com um serviço que PODE EXECUTAR ORDEM REAL, se houver
*assignment* ativo para o símbolo enviado. `ENABLE_REAL_TRADING` continua
bloqueado por design (ADR 0001) e este módulo não decide POR CONTA PRÓPRIA
quando enviar um sinal — é fundação testável, como o resto de `execution/`
(ver `PaperBroker`, `RiskManager`): quem orquestra decide se e quando chamar
`enviar()`, e hoje ninguém no pipeline de produção o faz.

Gaps conhecidos do contrato (não resolvidos por este módulo)
--------------------------------------------------------------
- **Sem endpoint de reconciliação conhecido.** O contrato compartilhado só
  cobre ENVIO (`POST /v1/signals`, resposta 202 fire-and-forget). G1 do ADR
  0014 (comparar posição nossa vs. posição real na corretora) segue aberto
  até existir uma forma de CONSULTAR fills/posições no relay.
- **HTTP, não HTTPS.** A `X-API-Key` viaja em texto puro pela rede — fora do
  controle deste código, é característica do endpoint do provedor.
- **`execution.entry/stopLoss/takeProfits` são só INFORMATIVOS.** O próprio
  contrato diz: "o parâmetro de execução efetivo é determinado pela
  configuração da estratégia no lado da plataforma quando esta os define."
  Enviar SL/TP aqui é uma sugestão, não uma garantia de que valem.

Idempotência (G5 do ADR 0014, do lado do relay)
-------------------------------------------------
`signalId` é a chave de idempotência do PROVEDOR — reenviar o mesmo id
devolve `{"status":"duplicate"}` sem nova ordem. `build_signal_id()` gera um
id determinístico (provider+symbol+timeframe+timestamp da barra), como o
contrato recomenda, para que um retry de rede seja seguro.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Protocol

import requests
from loguru import logger

_ENDPOINT = "/v1/signals"
_TIMEOUT_PADRAO = 10.0


class SignalType(str, Enum):
    ENTRY = "ENTRY"
    EXIT = "EXIT"
    CLOSE_PARTIAL = "CLOSE_PARTIAL"
    MOVE_SL = "MOVE_SL"
    MOVE_TP = "MOVE_TP"
    TRAILING_START = "TRAILING_START"


class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class AssetClass(str, Enum):
    CRYPTO = "CRYPTO"
    FOREX = "FOREX"
    STOCK = "STOCK"
    FUTURES = "FUTURES"
    OPTIONS = "OPTIONS"
    INDEX = "INDEX"


class EntryMode(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    ZONE = "ZONE"


class StopLossType(str, Enum):
    PRICE = "PRICE"
    PERCENT = "PERCENT"


class Strength(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"


class RoteamentoStatus(str, Enum):
    """Os quatro status do contrato + um nosso, para resposta fora do padrão."""

    ROUTED = "routed_to_assignments"
    DUPLICATE = "duplicate"
    IGNORED_TTL = "ttl_expired"
    IGNORED_NO_ASSIGNMENT = "no_active_assignment_for_symbol"
    ERRO_INESPERADO = "erro_inesperado"  # não é do contrato — nosso catch-all


@dataclass(frozen=True)
class TakeProfit:
    name: str
    price: Decimal
    close_percent: Decimal = Decimal("100")


@dataclass(frozen=True)
class TradingSignal:
    """
    Um sinal no formato que o relay espera. Campos opcionais de execução
    (stop_loss, take_profits, contracts...) ficam None quando não definidos —
    ver aviso no docstring do módulo sobre eles serem só informativos.
    """

    signal_id: str
    provider: str
    symbol: str
    asset_class: AssetClass
    side: Side
    entry_mode: EntryMode = EntryMode.MARKET
    entry_price: Decimal | None = None
    exchange: str = "B3"
    timeframe: str = "D1"
    signal_type: SignalType = SignalType.ENTRY
    strategy: str | None = None
    ttl_seconds: int = 60
    stop_loss: Decimal | None = None
    stop_loss_type: StopLossType = StopLossType.PRICE
    take_profits: tuple[TakeProfit, ...] = ()
    contracts: Decimal | None = None
    confidence: int | None = None  # 0-100
    strength: Strength | None = None
    reason: str | None = None
    tags: tuple[str, ...] = ()
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RespostaEnvio:
    status: RoteamentoStatus
    http_status: int
    raw: dict[str, Any]

    @property
    def foi_roteado(self) -> bool:
        return self.status == RoteamentoStatus.ROUTED


def build_signal_id(provider: str, symbol: str, timeframe: str, bar_timestamp: datetime) -> str:
    """
    Id determinístico — o contrato recomenda isto justamente para que um
    RETRY de rede após falha seja seguro: reenviar o mesmo id vira
    `duplicate` no relay, nunca uma segunda ordem.
    """
    ts = bar_timestamp.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"{provider}-{symbol}-{timeframe}-{ts}"


def to_payload(sig: TradingSignal, *, agora: datetime | None = None) -> dict[str, Any]:
    """Monta o JSON exatamente no formato do contrato. Função pura — sem rede."""
    agora = agora or datetime.now(tz=UTC)
    payload: dict[str, Any] = {
        "signalId": sig.signal_id,
        "provider": sig.provider,
        "timing": {
            "createdAt": agora.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "createdAtMs": int(agora.timestamp() * 1000),
            "ttlSeconds": sig.ttl_seconds,
        },
        "market": {
            "symbol": sig.symbol,
            "exchange": sig.exchange,
            "assetClass": sig.asset_class.value,
            "timeframe": sig.timeframe,
        },
        "action": {
            "signalType": sig.signal_type.value,
            "side": sig.side.value,
        },
        "execution": {
            "entry": {
                "mode": sig.entry_mode.value,
                "price": float(sig.entry_price) if sig.entry_price is not None else None,
                "minPrice": None,
                "maxPrice": None,
            },
            "trailingStop": {"enabled": False},
        },
    }
    if sig.strategy is not None:
        payload["strategy"] = sig.strategy
    if sig.stop_loss is not None:
        payload["execution"]["stopLoss"] = {
            "type": sig.stop_loss_type.value,
            "value": float(sig.stop_loss),
        }
    if sig.take_profits:
        payload["execution"]["takeProfits"] = [
            {"name": tp.name, "price": float(tp.price), "closePercent": float(tp.close_percent)}
            for tp in sig.take_profits
        ]
    if sig.contracts is not None:
        payload["execution"]["contracts"] = float(sig.contracts)
    if sig.confidence is not None or sig.strength is not None or sig.reason or sig.tags:
        payload["metadata"] = {
            "confidence": sig.confidence,
            "strength": sig.strength.value if sig.strength else None,
            "reason": sig.reason,
            "tags": list(sig.tags),
        }
    if sig.attributes:
        payload["attributes"] = dict(sig.attributes)
    return payload


class Transporte(Protocol):
    """Seam de teste — troca `requests` por um dublê sem tocar rede nenhuma."""

    def post(self, url: str, *, json: dict, headers: dict, timeout: float) -> Any: ...


def _interpretar_resposta(http_status: int, body: dict[str, Any]) -> RoteamentoStatus:
    status_str = body.get("status", "")
    reason = body.get("reason", "")
    if status_str == "routed_to_assignments":
        return RoteamentoStatus.ROUTED
    if status_str == "duplicate":
        return RoteamentoStatus.DUPLICATE
    if status_str == "ignored" and reason == "ttl_expired":
        return RoteamentoStatus.IGNORED_TTL
    if status_str == "ignored" and reason == "no_active_assignment_for_symbol":
        return RoteamentoStatus.IGNORED_NO_ASSIGNMENT
    logger.warning(
        "SignalGateway: resposta fora do contrato conhecido (http={}): {}",
        http_status, body,
    )
    return RoteamentoStatus.ERRO_INESPERADO


class SignalGatewayClient:
    """
    Cliente do relay de sinais. NÃO decide quando enviar — só envia o que
    mandarem, e devolve o veredito do relay já interpretado.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        transporte: Transporte | None = None,
        timeout: float = _TIMEOUT_PADRAO,
    ) -> None:
        if not base_url or not api_key:
            raise ValueError(
                "SignalGatewayClient exige base_url e api_key configurados — "
                "sem credencial este cliente se recusa a existir (ver "
                "MT5_SIGNAL_BASE_URL / MT5_SIGNAL_API_KEY, nunca literal em código)"
            )
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._transporte: Transporte = transporte if transporte is not None else requests

    def enviar(self, sig: TradingSignal) -> RespostaEnvio:
        """
        Envia UM sinal. Erros de rede/timeout NÃO são engolidos — sinal de
        saída que falha em silêncio é pior que um traceback (mesma doutrina
        ORA-006 do resto do projeto: fallback pode existir, invisível não).

        Hard-stop (SIGNAL_GATEWAY_HARDSTOP01, Regra 1 do CLAUDE.md): este
        método pode disparar ordem real do lado do relay (assignment ativo
        no símbolo) e por isso nunca deve prosseguir sem checar
        `settings.enable_real_trading` aqui dentro — defesa em profundidade,
        não só na validação de `Settings` no boot (que hoje já impede
        `enable_real_trading=True`, mas `enviar()` não deve confiar só nisso).
        """
        from config import settings

        if settings.enable_real_trading is not True:
            raise RuntimeError(
                "SignalGatewayClient.enviar() bloqueado: ENABLE_REAL_TRADING "
                "não está explicitamente True. Nenhuma ordem real é permitida "
                "por design (ver docs/decisions/0001-no-real-trading.md)."
            )

        payload = to_payload(sig)
        url = f"{self.base_url}{_ENDPOINT}"
        try:
            resp = self._transporte.post(
                url, json=payload,
                headers={"X-API-Key": self.api_key, "Content-Type": "application/json"},
                timeout=self.timeout,
            )
        except requests.exceptions.RequestException:
            logger.error("SignalGateway: falha de rede enviando {}", sig.signal_id)
            raise

        try:
            body = resp.json()
        except ValueError:
            body = {}
            logger.warning(
                "SignalGateway: resposta http={} sem corpo JSON válido", resp.status_code,
            )

        status = _interpretar_resposta(resp.status_code, body)
        return RespostaEnvio(status, resp.status_code, body)


def from_settings(*, timeout: float = _TIMEOUT_PADRAO) -> SignalGatewayClient:
    """Constrói o cliente a partir de `config.settings` — nunca de valor literal."""
    from config import settings

    return SignalGatewayClient(
        settings.mt5_signal_base_url, settings.mt5_signal_api_key, timeout=timeout,
    )
