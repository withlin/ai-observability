import json
from typing import Annotated

from langchain_core.tools import InjectedToolCallId

from context import current_context
from telemetry import Telemetry

# Rates are quoted per 1 USD; cross rates are derived from this table.
USD_RATES = {
    "USD": 1.0,
    "EUR": 0.92,
    "JPY": 150.0,
    "GBP": 0.79,
    "CAD": 1.36,
    "CNY": 7.20,
    "AUD": 1.52,
    "CHF": 0.88,
}

WEATHER = {
    "tokyo": "Tokyo: clear, +24°C",
    "lisbon": "Lisbon: partly cloudy, +21°C",
    "toronto": "Toronto: cloudy, +15°C",
}


def _weather_for(city: str) -> str:
    name = city.strip()
    return WEATHER.get(name.lower(), f"{name.title()}: clear, +20°C")


def _rate_data(source: str) -> dict:
    base = USD_RATES.get(source)
    if base is None:
        return {"result": "error", "error-type": "unknown-code"}
    return {
        "result": "success",
        "rates": {
            code: rate / base for code, rate in USD_RATES.items()
        },
    }


class WebTools:
    """Weather and currency tools bound to their owning agents."""

    def __init__(self, telemetry: Telemetry,
                 weather_agent: str = "weather_agent",
                 currency_agent: str = "currency_agent"):
        self._telemetry = telemetry
        self._weather_agent = weather_agent
        self._currency_agent = currency_agent

    def weather_tool(self):
        def get_weather(city: str,
                        tool_call_id: Annotated[str, InjectedToolCallId] = "") -> str:
            """Get the current weather for a city (simulated data)."""
            context = current_context()
            with self._telemetry.tool_span(
                "get_weather",
                "execute_tool get_weather",
                tool_call_id,
                {"city": city},
                description=get_weather.__doc__,
                agent_name=self._weather_agent,
                context=context,
            ) as span:
                result = _weather_for(city)
                span.set_result(result)
                return result

        return get_weather

    def currency_tool(self):
        def convert_currency(amount: float, from_currency: str, to_currency: str,
                             tool_call_id: Annotated[str, InjectedToolCallId] = "") -> str:
            """Convert an amount between currencies using simulated rates."""
            context = current_context()
            arguments = {
                "amount": amount,
                "from_currency": from_currency,
                "to_currency": to_currency,
            }
            with self._telemetry.tool_span(
                "convert_currency",
                "execute_tool convert_currency",
                tool_call_id,
                arguments,
                description=convert_currency.__doc__,
                agent_name=self._currency_agent,
                context=context,
            ) as span:
                source = from_currency.strip().upper()
                target = to_currency.strip().upper()
                data = _rate_data(source)
                if data.get("result") != "success":
                    raise RuntimeError(
                        f"rate provider error: {data.get('error-type', 'unknown')}"
                    )
                rates = data.get("rates") or {}
                if target not in rates:
                    raise KeyError(f"unknown target currency {target!r}")
                rate = float(rates[target])
                result = json.dumps({
                    "amount": float(amount),
                    "from": source,
                    "to": target,
                    "rate": rate,
                    "converted": round(float(amount) * rate, 2),
                })
                span.set_result(result)
                return result

        return convert_currency
