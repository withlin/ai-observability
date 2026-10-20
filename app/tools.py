import concurrent.futures
import json
import urllib.parse
import urllib.request
from typing import Annotated

from langchain_core.tools import InjectedToolCallId

from context import current_context
from telemetry import Telemetry


class HttpClient:
    """Text fetcher with a wall-clock deadline that also covers DNS/connect stalls."""

    def __init__(self, timeout: float):
        self._timeout = timeout
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="tool-http"
        )

    def _fetch(self, url: str) -> str:
        request = urllib.request.Request(url, headers={"User-Agent": "curl/8.0"})
        with urllib.request.urlopen(request, timeout=self._timeout) as response:
            return response.read().decode("utf-8", "replace")

    def get_text(self, url: str) -> str:
        future = self._pool.submit(self._fetch, url)
        try:
            return future.result(timeout=self._timeout + 5)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(
                f"HTTP fetch exceeded {self._timeout + 5:.0f}s: {url}"
            ) from None

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


class WebTools:
    """Weather and currency tools bound to their owning agents."""

    def __init__(self, http: HttpClient, telemetry: Telemetry,
                 weather_agent: str = "weather_agent",
                 currency_agent: str = "currency_agent"):
        self._http = http
        self._telemetry = telemetry
        self._weather_agent = weather_agent
        self._currency_agent = currency_agent

    def weather_tool(self):
        def get_weather(city: str,
                        tool_call_id: Annotated[str, InjectedToolCallId] = "") -> str:
            """Get the current weather for a city (real-time, via wttr.in)."""
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
                result = self._http.get_text(
                    "https://wttr.in/" + urllib.parse.quote(city.strip()) + "?format=3"
                ).strip()
                span.set_result(result)
                return result

        return get_weather

    def currency_tool(self):
        def convert_currency(amount: float, from_currency: str, to_currency: str,
                             tool_call_id: Annotated[str, InjectedToolCallId] = "") -> str:
            """Convert an amount between currencies using real-time rates (via open.er-api.com)."""
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
                data = json.loads(self._http.get_text(
                    f"https://open.er-api.com/v6/latest/{urllib.parse.quote(source)}"
                ))
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
