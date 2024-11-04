import io
import time
import json
import pickle


from src.utils import *
import torch
import torch.nn as nn


BUFFER_SIZE = 1024
HEADERSIZE = 10


model_config = {
    "actor": {
        "name": "ActorNetwork",
        "state_size": 29
              },
    "critic": {
        "name": "CriticNetwork",
        "state_size": 29,
        "action_size": 3
               }
    }

init_config = {"init_type": "xavier",
               "init_gain": 1.0,
               "gpu_ids": []}


class Client():
    def __init__(self, clientId, conn, address):
        self.clientId = clientId
        self.conn = conn
        self.address = address
        #byte array of files
        self.actormodelfile = None
        self.criticmodelfile = None

    def getClientId(self):
        return self.clientId

    def getConn(self):
        return self.conn

    def getAddress(self):
        return self.address

    def saveActorModelFile(self, file):
        self.actormodelfile = file

    def getActorModelFile(self):
        return self.actormodelfile

    def saveCriticModelFile(self, file):
        self.criticmodelfile = file

    def getCriticModelFile(self):
        return self.criticmodelfile

    def transmitModels(self, round):
        try:
            actorModelIO = bytes(self.actormodelfile)
            criticModelIO = bytes(self.criticmodelfile)
            load = {"actormodel": actorModelIO,
                    "criticmodel": criticModelIO,
                    "type": "models",
                    "round": round}
            serialised_data = pickle.dumps(load)

            serialised_data = bytes(
                f"{len(serialised_data):<{HEADERSIZE}}", 'utf-8') + serialised_data
            self.conn.send(serialised_data)
            return True
        except Exception as e:
            return False

    def receiveModels(self, round):
        modelData = b''
        timeout = 10  # [seconds 15
        timeout_start = time.time()
        flag = False
        firstByte = True

        try:
            while time.time() < timeout_start + timeout:
                data = self.conn.recv(BUFFER_SIZE)
                if firstByte:
                    # print("new msg len:", data[:HEADERSIZE])
                    msglen = int(data[:HEADERSIZE])
                    firstByte = False

                modelData += data

                if len(modelData)-HEADERSIZE == msglen:
                    # print(modelData[HEADERSIZE:])
                    # print(pickle.loads(modelData[HEADERSIZE:]))
                    flag = True
                    break

        except Exception as e:
            print("")

        if flag is True:
            unpickle_data = pickle.loads(modelData[HEADERSIZE:])

            actorModelIO = io.BytesIO(bytes(unpickle_data["actormodel"]))
            criticModelIO = io.BytesIO(bytes(unpickle_data["criticmodel"]))

            return actorModelIO, criticModelIO
        else:
            return None, None
            return None, None
