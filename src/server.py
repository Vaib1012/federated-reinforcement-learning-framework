import copy
import gc
import logging
import threading
import numpy as np
import torch
import torch.nn as nn
import time

from multiprocessing import pool, cpu_count
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from collections import OrderedDict

from src.models import *
from src.utils import *
from src.Connection import *


logger = logging.getLogger(__name__)

BUFFER_SIZE = 1024


class Server(object):
    """Class for implementing center server orchestrating the whole process of federated learning

    At first, center server distribute model skeleton to all participating clients with configurations.
    While proceeding federated learning rounds, the center server samples some fraction of clients,
    receives locally updated parameters, averages them as a global parameter (model), and apply them to global model.
    In the next round, newly selected clients will recevie the updated global model as its local model.
    """

    def __init__(self, writer, model_config={}, global_config={}, init_config={}, fed_config={}):
        self.clients = None
        self._round = 0
        self.writer = writer

        self.model_config = model_config
        self.actormodel = eval(model_config["actor"]["name"])(
            **model_config["actor"])

        self.criticmodel = eval(model_config["critic"]["name"])(
            **model_config["critic"])

        self.seed = global_config["seed"]
        self.device = global_config["device"]
        self.mp_flag = global_config["is_mp"]

        self.init_config = init_config

        self.fraction = fed_config["C"]
        self.num_clients = fed_config["K"]
        self.num_rounds = fed_config["R"]
        self.local_epochs = fed_config["E"]
        self.batch_size = fed_config["B"]

        self.actormodelPath = fed_config["actormodel_path"]
        self.criticmodelPath = fed_config["criticmodel_path"]

        self.conn = startTCPServer()

    def setup(self, **init_kwargs):
        """Set up all configuration for federated learning."""

        # valid only before the very first round
        assert self._round == 0

        # initialize weights of the model
        torch.manual_seed(self.seed)
        init_net(self.actormodel, **self.init_config)
        init_net(self.criticmodel, **self.init_config)

        message = f"[Round: {str(self._round).zfill(4)}] ...successfully initialized actor model (# parameters: {str(sum(p.numel() for p in self.actormodel.parameters()))})!"
        self.logMessage(message)

        message = f"[Round: {str(self._round).zfill(4)}] ...successfully initialized critic model (# parameters: {str(sum(p.numel() for p in self.criticmodel.parameters()))})!"
        self.logMessage(message)

        forward = input(" enter any no to proceed : ")
        clientsConnections = getAllClients()
        self.clients = (clientsConnections)
        # send the model skeleton to all clients
        # self.transmit_model()

    def transmit_model(self, sampled_client_indices=None):
        """Send the updated global model to selected/all clients."""
        self.actormodel.load_state_dict(torch.load(
            self.actormodelPath, map_location=torch.device('cpu')))
        self.criticmodel.load_state_dict(torch.load(
            self.criticmodelPath, map_location=torch.device('cpu')))

        actormodelFile = open(self.actormodelPath, 'rb')
        criticmodelFile = open(self.criticmodelPath, 'rb')
        actormodelFileByteArray = self.convertFileToByteArray(actormodelFile)
        criticmodelFileByteArray = self.convertFileToByteArray(criticmodelFile)
        clientIndices = []

        if sampled_client_indices is None:
            # send the global model to all clients before the very first and after the last federated round
            assert (self._round == 0) or (self._round == self.num_rounds)
            for client in tqdm(self.clients, leave=False):
                client.saveActorModelFile(copy.deepcopy(
                    actormodelFileByteArray))
                client.saveCriticModelFile(copy.deepcopy(
                    criticmodelFileByteArray))

            for i in range(0, len(self.clients)):
                clientIndices.append(i)

            message = f"[Round: {str(self._round).zfill(4)}] ...successfully copied models to all {str(len(self.clients))} client instances!"
            self.logMessage(message)

        else:
            # send the global model to selected clients
            assert self._round != 0
            # print("index---------------------" + str(sampled_client_indices))
            for idx in sampled_client_indices:
                self.clients[idx].saveActorModelFile(copy.deepcopy(
                    actormodelFileByteArray))
                self.clients[idx].saveCriticModelFile(copy.deepcopy(
                    criticmodelFileByteArray))
            clientIndices = sampled_client_indices

            message = f"[Round: {str(self._round).zfill(4)}] ...successfully copied models to selected {str(self.num_clients)} client instances!"
            self.logMessage(message)
            # clientIndices = sampled_client_indices

        """Multiprocessing-applied version of "update_selected_clients" method."""
        with pool.ThreadPool(processes=len(clientIndices)) as workhorse:
            workhorse.map(self.send_model_to_clients, clientIndices)

        # for i in clientIndices:
        #     self.send_actormodel_to_clients(i)

        # for i in clientIndices:
        #     self.send_criticmodel_to_clients(i)

        # print(result)

    def collectModelsFromSelectedClients(self, sampled_client_indices):
        with pool.ThreadPool(processes=len(sampled_client_indices)) as workhorse:
            models = workhorse.map(
                self.collectModelClient, sampled_client_indices)
        # print("\n\n\n\--------------printing received models ")
        # print(models)

        # print("\n ROUN ")

        actormodels = []
        criticmodels = []
        for model in models:
            if model[0] is not None:
                actormodels.append(model[0])
            if model[1] is not None:
                criticmodels.append(model[1])
        message = f"[Round: {str(self._round).zfill(4)}] ...models received from {str(len(models))} clients"
        self.logMessage(message)
        return actormodels, criticmodels

    def collectModelClient(self, selectedClientInd):
        clientActorModel = None
        clientCriticModel = None
        # receive actor model from the client
        client_id = self.clients[selectedClientInd].getClientId()

        message = f"[Round: {str(self._round).zfill(4)}] ...waiting for client {client_id.zfill(4)} to send model updates"
        self.logMessage(message)
        actormodelIO, criticmodelIO = self.clients[selectedClientInd].receiveModels(
            self._round)

        if(actormodelIO is not None and criticmodelIO is not None):
            clientActorModel = eval(self.model_config["actor"]["name"])(
                **self.model_config["actor"])
            init_net(clientActorModel, **self.init_config)
            state_dict = torch.load(
                actormodelIO, map_location=torch.device('cpu'))
            clientActorModel.load_state_dict(state_dict)

            clientCriticModel = eval(self.model_config["critic"]["name"])(
                **self.model_config["critic"])
            init_net(clientActorModel, **self.init_config)
            state_dict = torch.load(
                criticmodelIO, map_location=torch.device('cpu'))
            clientCriticModel.load_state_dict(state_dict)

            message = f"[Round: {str(self._round).zfill(4)}] ...client {client_id.zfill(4)} successfully received actor and critic models"
        else:
            # message = f"[Round: {str(self._round).zfill(4)}] ...client {client_id.zfill(4)} didnt receive actor model updates"
            message = f"[Round: {str(self._round).zfill(4)}] ...client {client_id.zfill(4)} successfully received actor and critic models"

        self.logMessage(message)

        return clientActorModel, clientCriticModel

    def send_model_to_clients(self, selectedClientInd):
        # update selected clients
        client_id = str(self.clients[selectedClientInd].getClientId())
        message = f"[Round: {str(self._round).zfill(4)}] Start sending file to the selected client {client_id.zfill(4)}...!"
        self.logMessage(message)

        # transmit actor model to a client
        result = self.clients[selectedClientInd].transmitModels(self._round)
        if(result):
            message = f"[Round: {str(self._round).zfill(4)}] ...client {client_id.zfill(4)} file successfully transmitted actor and critic models"
        else:
            message = f"[Round: {str(self._round).zfill(4)}] ...client {client_id.zfill(4)}  transmitting actor and critic models"
        self.logMessage(message)

    def fit(self):
        """Execute the whole process of the federated learning."""
        # self.results = {"reward": [], "damage": [], "steps": []}
        for r in range(self.num_rounds):
            self._round = r + 1

            self.train_federated_model()

            print("\n\t ROUND : "+str(self._round) + " COMPLETED...!")
            time.sleep(5)

    def train_federated_model(self):
        """Do federated training."""
        # select pre-defined fraction of clients randomly
        # sampled_client_indices = self.sample_clients()
        sampled_client_indices = [x for x in range(len(self.clients))]
        print("\n\n sample")
        print(sampled_client_indices)
        # send global model to the selected clients
        self.transmit_model(sampled_client_indices)
        time.sleep(4)  # 20
        # list of collected models
        actorModels, criticmodels = self.collectModelsFromSelectedClients(
            sampled_client_indices)

        coefficient = 1.0 / len(self.clients)
        self.saveActorModel(self.average_model(actorModels, coefficient))
        self.saveCriticModel(self.average_model(criticmodels, coefficient))

    def average_model(self, models, coefficient):
        """Average the updated and transmitted parameters from each selected client."""
        averaged_weights = OrderedDict()
        it = 0
        for model in models:
            if model is not None:
                # get model updates here
                local_weights = model.state_dict()

                for key in model.state_dict().keys():
                    if it == 0:
                        averaged_weights[key] = coefficient * \
                            local_weights[key]
                    else:
                        averaged_weights[key] += coefficient * \
                            local_weights[key]

        return averaged_weights

    def saveActorModel(self, averaged_weights):
        try:
            message = f"[Round: {str(self._round).zfill(4)}] ...{str(self.actormodel)}"
            self.logMessage(message)

            self.actormodel.load_state_dict(averaged_weights)
            # save model
            torch.save(self.actormodel.state_dict(),
                       "./models/aggregated_actor.pth")

        except Exception as e:
            # message = f"[Round: {str(self._round).zfill(4)}] ...error occured while aggregating actor models"
            # self.logMessage(message)
            print("")

    def saveCriticModel(self, averaged_weights):
        try:
            message = f"[Round: {str(self._round).zfill(4)}] ...{str(self.criticmodel)}"
            self.logMessage(message)
            self.criticmodel.load_state_dict(averaged_weights)
            # save model
            torch.save(self.criticmodel.state_dict(),
                       "./models/aggregated_critic.pth")

        except:
            # message = f"[Round: {str(self._round).zfill(4)}] ...error occured while aggregating critic models"
            # self.logMessage(message)
            print("")

    def sample_clients(self):
        """Select some fraction of all clients."""
        # sample clients randommly
        message = f"[Round: {str(self._round).zfill(4)}] Select clients...!"
        self.logMessage(message)

        clientsConnected = len(self.clients)
        num_sampled_clients = max(int(self.fraction * clientsConnected), 1)
        sampled_client_indices = sorted(np.random.choice(a=[i for i in range(
            self.num_clients)], size=num_sampled_clients, replace=False).tolist())

        return sampled_client_indices

    def logMessage(self, message):
        print(message)
        logging.info(message)
        del message
        gc.collect()

    def convertFileToByteArray(self, file):
        byteArrOfFile = []
        while True:
            line = file.read(BUFFER_SIZE)
            while (line):
                byteArrOfFile.extend(line)
                line = file.read(BUFFER_SIZE)
            if not line:
                file.close()
                break
        return byteArrOfFile
