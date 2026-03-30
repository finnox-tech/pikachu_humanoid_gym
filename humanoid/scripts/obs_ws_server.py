import asyncio
import json
import logging

import websockets

logging.basicConfig(level=logging.INFO)

VIEWERS = set()

async def handler(websocket, path):
    logging.info(f"Client connected: {websocket.remote_address}")
    try:
        async for message in websocket:
            try:
                obj = json.loads(message)
            except json.JSONDecodeError:
                continue

            client_type = obj.get("client_type")
            if client_type == "viewer":
                VIEWERS.add(websocket)
                logging.info(f"Viewer registered: {websocket.remote_address} (total={len(VIEWERS)})")
                await websocket.send(json.dumps({"status": "viewer_registered"}))
                continue

            if client_type == "sim":
                logging.info(f"Sim client registered: {websocket.remote_address}")
                await websocket.send(json.dumps({"status": "sim_registered"}))
                continue

            if obj.get("source") == "sim":
                payload = obj.get("data")
                if payload is None:
                    continue

                dead_viewers = []
                for v in VIEWERS:
                    try:
                        await v.send(json.dumps(payload))
                    except Exception:
                        dead_viewers.append(v)
                for v in dead_viewers:
                    VIEWERS.discard(v)

    except websockets.exceptions.ConnectionClosed:
        logging.info(f"Client disconnected: {websocket.remote_address}")
        VIEWERS.discard(websocket)


def main():
    host = "localhost"
    port = 8765
    start_server = websockets.serve(handler, host, port)
    logging.info(f"Starting OBS WebSocket server on ws://{host}:{port}")
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()


if __name__ == "__main__":
    main()
