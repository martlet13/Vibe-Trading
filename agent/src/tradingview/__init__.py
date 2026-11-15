"""TradingView signal intake for AI Agent Lens packets.

AI Agent Lens (https://github.com/roan-maverick/tradingview-indicator-for-ai-agents,
MIT) is a TradingView indicator that posts one JSON packet per closed candle
to a webhook. This package receives those packets and keeps the newest one
per chart where the agent's ``tradingview_signals`` tool can read it.

The receiver is its own small HTTP server, not a route on the API server. A
webhook must be reachable from the internet through a tunnel, and the API
server trusts every loopback client: tunnelling its port would hand the whole
API to anyone who found the URL. The receiver exposes one POST endpoint,
requires a shared token, and only writes validated packets to disk.

A packet is indicator output from an outside service. Nothing here places an
order: the agent reads a packet as data, and any order it then proposes goes
through the normal trading tools and their mandate gate.
"""
