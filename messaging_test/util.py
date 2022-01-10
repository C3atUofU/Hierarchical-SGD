import socket
import pickle
import struct
import argparse


def send_msg(sock, msg):
    msg_pickle = pickle.dumps(msg)
    sock.sendall(struct.pack(">I", len(msg_pickle)))
    sock.sendall(msg_pickle)
    print(msg[0], 'sent to', sock.getpeername())


def recv_msg(sock, expect_msg_type = None):
    msg_len = struct.unpack(">I", sock.recv(4))[0]
    msg = sock.recv(msg_len, socket.MSG_WAITALL)
    msg = pickle.loads(msg)
    print(msg[0], 'received from', sock.getpeername())

    if (expect_msg_type is not None) and (msg[0] != expect_msg_type):
        raise Exception("Expected " + expect_msg_type + " but received " + msg[0])
    return msg


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ip', type=str, default='localhost', help='Server IP address')
    parser.add_argument('-port', type=int, default=51018, help='Server port')
    parser.add_argument('-size', type=int, default=132863336, help='Number of floating point parameters in message')
    parser.add_argument('-sim', type=int, default=10, help='Number of simulation rounds')

    args = parser.parse_args()
    return args