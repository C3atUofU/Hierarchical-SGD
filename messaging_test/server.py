import socket
from util import *

if __name__ == '__main__':
    args = args_parser()

    listening_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listening_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listening_sock.bind((args.ip, args.port))

    while True:
        listening_sock.listen(5)
        print("Waiting for incoming connection...")
        (client_sock, (ip, port)) = listening_sock.accept()
        print('Got connection from ', (ip, port))
        print(client_sock)

        try:
            m = recv_msg(client_sock, 'MSG_CLIENT_TO_SERVER')
            a = m[1]
            send_msg(client_sock, ['MSG_SERVER_TO_CLIENT', a])
        except Exception as e:
            print(e)

