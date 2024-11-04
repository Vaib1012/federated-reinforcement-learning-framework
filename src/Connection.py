import socket
import sys
import threading
import time
import random
from src.client import Client

clients = []
# all_address = []


def create_socket():
    try:
        global host
        global port
        global s
        host = "localhost"
        port = 9999
        s = socket.socket()

    except socket.error as msg:
        print("Socket creation error: " + str(msg))


# Binding the socket and listening for connections
def bind_socket():
    try:
        global host
        global port
        global s
        print("Binding the Port: " + str(port))

        s.bind((host, port))
        s.listen(10)

    except socket.error as msg:
        print("Socket Binding error" + str(msg) + "\n" + "Retrying...")
        bind_socket()


def getAllClients():
    return clients


def accepting_connections():
    global host
    global port
    global s
    while True:
        try:
            conn, address = s.accept()
            s.setblocking(1)  # prevents timeout
            id = f'rl_client-{random.randint(0, 1000)}'

            # create and append client object to the list
            client = Client(id, conn, address)
            clients.append(client)
            # print(conn.recv(1024).decode())
            # conn.send(" ".encode())
            print("Connection has been established :" + address[0])
            print("Active Clients: "+str(len(clients)))

        except:
            print("Error accepting connections")


# def removeInactiveCLients():
#     for idx in range(len(clients)):
#         if clients[idx].isActive() is False:
#             clients.remove(idx)


def startTCPServer():
    create_socket()
    bind_socket()
    acceptConnThread = threading.Thread(target=accepting_connections)
    acceptConnThread.daemon = True
    acceptConnThread.start()
