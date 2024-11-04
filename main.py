import yaml
import os
import threading
import logging
import pickle
import time

from torch.utils.tensorboard import SummaryWriter
from src.utils import launch_tensor_board

from src.models import *
from src.utils import *
from src.server import Server

if __name__ == "__main__":
    # read configuration file
    with open('./config.yaml') as c:
        configs = list(yaml.load_all(c, Loader=yaml.FullLoader))
    global_config = configs[0]["global_config"]
    fed_config = configs[1]["fed_config"]
    init_config = configs[2]["init_config"]
    model_config = configs[3]["model_config"]
    log_config = configs[4]["log_config"]

    # modify log_path to contain current time
    # log_config["log_path"] = os.path.join(log_config["log_path"], str(
    #     datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))
    log_config["log_path"] = os.path.join(log_config["log_path"], str("FL"))

    # initiate TensorBaord for tracking losses and metrics
    writer = SummaryWriter(
        log_dir=log_config["log_path"], filename_suffix="FL")

    # tensorborad thread
    # tb_thread = threading.Thread(
    #     target=launch_tensor_board,
    #     args=([log_config["log_path"], log_config["tb_port"], log_config["tb_host"]])
    #     ).start()

    # set the configuration of global logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=os.path.join(log_config["log_path"], log_config["log_name"]),
        level=logging.INFO,
        format="[%(levelname)s](%(asctime)s) %(message)s",
        datefmt="%Y/%m/%d/ %I:%M:%S %p")

    # display and log experiment configuration
    message = "\n[WELCOME] Unfolding configurations...!"
    print(message)
    logging.info(message)

    # for config in configs:
    #     print(config)
    #     logging.info(config)
    # print()

    # initialize federated learning
    central_server = Server(writer, model_config, global_config,
                            init_config, fed_config)

    central_server.setup()

    # do federated learning
    central_server.fit()

    # # save resulting losses and metrics
    # with open(os.path.join(log_config["log_path"], "result.pkl"), "wb") as f:
    #     pickle.dump(central_server.results, f)

    # bye!
    message = "...done all learning process!\n...exit program!"
    print(message)
    logging.info(message)
    time.sleep(3)
    exit()
