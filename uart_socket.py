"""
# ターミナル1（サーバ）
python uart_socket.py server --port 7000

# ターミナル2（クライアント）
python uart_socket.py client --host localhost --port 7000

"""

# uart_socket.py
import argparse
import time
import serial

def run_server(port=7000):
    # サーバ（エコーバック）
    ser = serial.serial_for_url(f"socket://:{port}?server=1", timeout=1)
    print(f"[SERVER] listening on TCP {port}")
    try:
        while True:
            data = ser.read(1024)  # 任意バイナリOK
            if data:
                print("[SERVER] RX:", data)
                ser.write(data)     # そのまま返す（エコー）
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()

def run_client(host="localhost", port=7000):
    ser = serial.serial_for_url(f"socket://{host}:{port}", timeout=1)
    print(f"[CLIENT] connected to {host}:{port}")
    try:
        for i in range(5):
            msg = f"hello {i}\n".encode()
            print("[CLIENT] TX:", msg)
            ser.write(msg)
            time.sleep(0.2)
            rx = ser.readline()
            if rx:
                print("[CLIENT] ECHO:", rx.rstrip())
        print("[CLIENT] done")
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    ap_srv = sub.add_parser("server")
    ap_srv.add_argument("--port", type=int, default=7000)

    ap_cli = sub.add_parser("client")
    ap_cli.add_argument("--host", default="localhost")
    ap_cli.add_argument("--port", type=int, default=7000)

    args = ap.parse_args()
    if args.mode == "server":
        run_server(args.port)
    else:
        run_client(args.host, args.port)
