import socket
from util import *
import numpy as np
import time

if __name__ == '__main__':
    args = args_parser()


    time_send_list = []
    time_receive_list = []
    time_total_list = []


    for sim in range(0, args.sim):
        time.sleep(10)

        print('\n--- Simulation round', sim)

        array = np.random.randn(args.size).astype('float32')
        sock = socket.socket()
        sock.connect((args.ip, args.port))

        start_time = time.time()
        send_msg(sock, ['MSG_CLIENT_TO_SERVER', array])
        send_time = time.time() - start_time
        start_time2 = time.time()
        r = recv_msg(sock, 'MSG_SERVER_TO_CLIENT')
        receive_time = time.time() - start_time2

        sock.close()

        total_time = time.time() - start_time

        time_send_list.append(send_time)
        time_receive_list.append(receive_time)
        time_total_list.append(total_time)

        print('sending time of this round:', send_time)
        print('receiving time of this round:', receive_time)
        print('total time of this round:', total_time)

        print('*** aggregated statistics until simulation round', sim)
        print('average sending time:', np.mean(time_send_list))
        print('standard deviation of sending time:', np.std(time_send_list))
        print('average receiving time:', np.mean(time_receive_list))
        print('standard deviation of receiving time:', np.std(time_receive_list))
        print('average total time:', np.mean(time_total_list))
        print('standard deviation of total time:', np.std(time_total_list))




