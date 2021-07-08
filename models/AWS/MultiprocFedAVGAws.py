# importing libraries
import paho.mqtt.client as paho
import os
import psutil
import io
import h5py
import socket
import ssl
from tensorflow import keras
import random
import string
import pandas as pd
import numpy as np
import tensorflow as tf
import math
import json
from time import sleep
from random import uniform
from sklearn.metrics import accuracy_score
import sys
from sys import getsizeof
import multiprocessing


def global_vars(num_clients):
    '''
    Initialize all the global variables
    '''

    global clientKeys
    global upLimit
    global local_models_weight_list
    global received_clients
    global bytearrays_list
    global RondaComm
    global NumberSlicesClients
    global publish_flag
    global model
    global ACK
    global acc_results_list
    global loss_results_list

    loss_results_list = []
    acc_results_list = []
    publish_flag = False
    ACK = False
    NumberSlicesClients = 0
    clientKeys = 0
    upLimit = 0
    RondaComm = 0
    local_models_weight_list = []
    received_clients = [int()] * (num_clients + 1) * 200
    bytearrays_list = [bytearray()] * (num_clients + 1) * 200


def clean_vars():
    '''
    Clean the variables at each comm round
    '''

    global upLimit
    global received_clients
    global bytearrays_list
    global local_models_weight_list

    local_models_weight_list = []
    upLimit = 0
    received_clients = [int()] * (num_clients + 1) * 200
    bytearrays_list = [bytearray()] * (num_clients + 1) * 200


def initialize_topic(num_clients):
    '''
    Get the array of all the topics of the subscriptions
    '''

    topic = "home/ModelRP"
    topic_list = []
    for i in range(1, num_clients + 1):
        print(topic + str(i))
        topic_list.append(topic + str(i))

    return topic_list


def fit_model(model):
    '''
    Function for model training
    '''

    model.fit(x_train, y_train, epochs=5, verbose=1)
    model.save('SavedModel.h5')
    return model


def define_model():
    '''
    Function for model creation
    '''

    print("Creating model")
    input_layer = tf.keras.layers.Flatten(input_shape=(784,))
    dense_layer_1 = tf.keras.layers.Dense(128, activation='relu')
    dense_layer_2 = tf.keras.layers.Dropout(0.2)
    output = tf.keras.layers.Dense(10, activation='softmax')
    model = tf.keras.models.Sequential([
        input_layer,
        dense_layer_1,
        dense_layer_2,
        output
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def setup_data():
    '''
    Extract the data from the dataset
    '''

    data = np.load('./mnist1.npz')
    x_train = data['x_train']
    x_test = data['x_test']
    y_train = data['y_train']
    y_test = data['y_test']

    # Preprocessing
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Add one domention to make 3D images
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    return x_train, x_test, y_test, y_train


def get_model(payload):
    '''
    Construct the model from the bytearray
    '''

    f = io.BytesIO(payload)
    try:
        h = h5py.File(f, 'r')
        new_model = keras.models.load_model(h)
        f.close()
        return new_model
    except:  # catch *all* exceptions
        e = sys.exc_info()[0]
        print("Error: " + str(e))
        return False


def model_testing(model):
    '''
    Get the accuracy of the global model
    '''
    global acc_results_list
    global loss_results_list

    eval_loss, eval_acc = model.evaluate(x_test,  y_test, verbose=1)
    print('Eval accuracy percentage: {:.2f}'.format(eval_acc * 100))

    acc_results_list.append(round(eval_acc*100, 2))
    loss_results_list.append(round(eval_loss*100, 2))

    f = open("logs/server_acc.txt", "a")
    f.truncate(0)
    f.write(str(acc_results_list))
    f.close()

    f = open("logs/server_loss.txt", "a")
    f.truncate(0)
    f.write(str(loss_results_list))
    f.close()

    """     cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        #logits = model.predict(X_test, batch_size=100)
        logits = model.predict(x_test)
        loss = cce(y_test, logits)
        acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(y_test, axis=1))
        print('global_acc: {:.3%} | global_loss: {}'.format(acc, loss)) """
    """## Evaluation"""


def on_connect(client, userdata, flags, rc):
    '''
    Function for making connection 
    '''

    global connflag
    global clientKeys
    for x in range(0, num_clients):
        mqttc.subscribe(topic_list[x], qos=1)
        clientKeys = clientKeys + x + 1
    mqttc.subscribe("home/RPIC", qos=1)
    print("Connected to AWS")
    connflag = True
    print("Connection returned result: " + str(rc))


def get_lengths(model):
    '''
    Get the lengths of the model in order to publish every chunk
    :return sliceLength: the length of each chunk that is going to be publish, except the last one
    :return finalSlice: the length of the last chunk
    :return ByteArray: the whole model converted to bytearray
    :return arrayLength: the length of the bytearray
    :return slicesNumber: the number of chunks that the model is divided into
    '''
    model.save('SavedModel.h5')
    fo = open(filename, "rb")
    ByteArray = fo.read()
    arrayLength = len(ByteArray)
    slicesNumber = math.ceil(arrayLength/128000)
    sliceLength = math.ceil(arrayLength/slicesNumber)
    if (arrayLength % slicesNumber) == 0:  # Check if each piece is whole
        finalSlice = sliceLength
    else:
        resto = (arrayLength % slicesNumber)
        # If there is remainder, the last slice must be smaller
        finalSlice = sliceLength - slicesNumber + resto
    return sliceLength, finalSlice, ByteArray, arrayLength, slicesNumber


def publish_model(model, j):
    '''
    Publish the model to a topic
    '''

    sliceLength, finalSlice, ByteArray, arrayLength, slicesNumber = get_lengths(
        model)
    print("Publishing")
    mqttc.publish("home/AWSC" + str(j),  bytearray(str(slicesNumber), "UTF-8") + bytearray(str("/"), "UTF-8") + bytearray(
        str(RondaComm), "UTF-8") + bytearray(str("/"), "UTF-8") + bytearray(str(slicesNumber) + "*", "UTF-8") + ByteArray[arrayLength - finalSlice:arrayLength], qos=1)
    sleep(1)
    for i in range(slicesNumber-1, 0, -1):
        mqttc.publish("home/AWSC" + str(j), bytearray(str(i), "UTF-8") + bytearray(str("/"), "UTF-8") + bytearray(
            str(RondaComm), "UTF-8") + bytearray(str("/"), "UTF-8") + bytearray(str(slicesNumber) + "*", "UTF-8") + ByteArray[sliceLength*(i-1):sliceLength*i], qos=1)
        sleep(1)
    sleep(3)
    print("All chunks were sent to home/AWSC", j)


def scale_model_weights(weight, scalar):
    '''
    Function for scalating the weights of every model
    '''
    weight_final = []
    weight_num = len(weight)
    for i in range(weight_num):
        # each weight is multiplied by the scalar that
        weight_final.append(scalar * weight[i])
        # defines what percentage of the total data belongs to this customer
    return weight_final


def average_model(weights_list):
    '''
    Function to average the weights of the models
    '''
    avg_grad = list()
    for tuples in zip(*weights_list):
        # sum of elements of the tensor
        layer_mean = tf.math.reduce_sum(tuples, axis=0)
        avg_grad.append(layer_mean)  # the averaged weights are added

    return avg_grad


def message_values(payload, topic):
    '''
    Get the differents params needed from the message
    :return chunkID: the ID from the chunk, from 1 to the total number of chunks.
    :return rComm: the round of communication in which the message was sent
    :return NumberSlices: the number of chunks the model is divided into
    :return limite+1: to determine which part of the payload is meta information
    '''

    if ((len(topic)) % 2) == 0:
        clientID = int(topic[-2:])
    else:
        clientID = int(topic[-1])
    limite = 0
    for i in range(0, 20):
        if (chr(payload[i]).isdigit() or chr(payload[i]) == '/'):
            limite += 1
        else:
            break
    tmp = str(payload[0:limite].decode("utf-8"))
    tmp = tmp.split("/")
    chunkID = int(tmp[0])
    rComm = int(tmp[1])
    NumberSlicesClients = int(tmp[2])
    slotArray = (((clientID - 1)*NumberSlicesClients) + chunkID)

    return rComm, NumberSlicesClients, clientID, slotArray, (limite+1)


def check_if_complete(clientID):
    '''
    Function to check if every chunk of a client has arrived
    '''

    global bytearrays_list
    global NumberSlicesClients

    tmp = True
    for i in range((((clientID-1)*NumberSlicesClients)+1), ((clientID-1)*NumberSlicesClients)+1 + NumberSlicesClients):
        if received_clients[i] == 0:
            tmp = False

    return tmp


def on_message(client, userdata, msg):
    '''
    Function for receiving msgs
    '''

    global upLimit
    global local_models_weight_list
    global received_clients
    global bytearrays_list
    global ACK
    global RondaComm
    global NumberSlicesClients
    global publish_flag
    global model

    rComm, NumberSlicesClients, clientID, slotArray, limite = message_values(
        msg.payload, msg.topic)
    if rComm == (RondaComm):  # If the received message belongs to the appropriate comm round
        # If all the slots of a client were covered
        if received_clients[slotArray] == 0:
            ACK = True
            received_clients[slotArray] = 1
            bytearrays_list[slotArray] = msg.payload[limite:]
            if(check_if_complete(clientID)):  # If all the slots of a client were covered
                upLimit += 1
                aux_bytearray = bytearray()
                for i in range(((clientID-1)*NumberSlicesClients) + NumberSlicesClients,  ((clientID-1)*NumberSlicesClients), -1):
                    aux_bytearray[0:0] = bytearrays_list[i]
                print("Whole client " +
                      str(clientID) + " model arrived")
                model = get_model(aux_bytearray)
                if model == False:
                    print("Fail at model construction")
                else:
                    scaled_weights = scale_model_weights(
                        model.get_weights(), 1/num_clients)
                    local_models_weight_list.append(scaled_weights)


def getMAC(interface='eth0'):
    '''
    Return the MAC address of the specified interface
    '''

    try:
        str = open('/sys/class/net/%s/address' % interface).read()
    except:
        str = "00:00:00:00:00:00"
    return str[0:17]


def getEthName():
    '''
    Get name of the Ethernet interface
    '''

    try:
        for root, dirs, files in os.walk('/sys/class/net'):
            for dir in dirs:
                if dir[:3] == 'enx' or dir[:3] == 'eth':
                    interface = dir
    except:
        interface = "None"
    return interface


def worker(ret_value):
    sleep(300)
    ret_value.value = 1


if __name__ == "__main__":

    import logging
    logging.getLogger('tensorflow').disabled = True
    connflag = False
    filename = "SavedModel.h5"  # file to send
    num_clients = int(sys.argv[1])
    num_process = int(sys.argv[3])

    clean_session_State = True

    if int(sys.argv[2]) == 1:
        clean_session_State = False

    global_vars(num_clients)
    global publish_flag
    global model
    global ACK

    x_train, x_test, y_test, y_train = setup_data()

    from tensorflow.keras.layers import Input, Dense, Activation, Dropout
    from tensorflow.keras.models import Model
    # mqttc object
    mqttc = paho.Client(client_id="AWS", clean_session=clean_session_State)
    # assign on_connect func
    mqttc.on_connect = on_connect
    # assign on_message func
    mqttc.on_message = on_message
    #mqttc.on_log = on_log

    #### *Common for publ and subs* ####
    awshost = "a3f3tevlek9dbt-ats.iot.us-east-1.amazonaws.com"      # Endpoint
    awsport = 8883                                              # Port no.
    clientID = "PabloClient"                                     # Thing_Name
    thingName = "PabloClient"                                    # Thing_Name
    # Root_CA_Certificate_Name
    caPath = "/home/ubuntu/modelComm/root-ca.pem"
    # <Thing_Name>.cert.pem
    certPath = "/home/ubuntu/modelComm/certificate.pem.crt"
    # <Thing_Name>.private.key
    keyPath = "/home/ubuntu/modelComm/private.pem.key"

    mqttc.tls_set(caPath, certfile=certPath, keyfile=keyPath, cert_reqs=ssl.CERT_REQUIRED,
                  tls_version=ssl.PROTOCOL_TLSv1_2, ciphers=None)  # pass parameters

    # connect to aws server
    mqttc.connect(awshost, awsport, keepalive=60)

    ### *Till here* ###
    ethName = getEthName()
    ethMAC = getMAC(ethName)
    macIdStr = ethMAC
    publish_flag = True
    model = define_model()
    topic_list = initialize_topic(num_clients)
    mqttc.loop_start()                                          # Start the loop

    while 1 == 1:
        if connflag == True:
            if (ACK == True):
                # probar con sleep de 0.1
                sleep(0.2)
                ACK = False
            if publish_flag == True:
                ret_value = multiprocessing.Value("d", 0, lock=False)
                publish_flag = False
                print("------------------------- Ronda Comm - " +
                      str(RondaComm) + " ------------------------- ")
                RondaComm = RondaComm + 1
                for j in range(1, num_process + 1):
                    publish_model(model, j)
                clean_vars()
                for j in range(1, num_process + 1):
                    mqttc.publish(
                        "home/AWSF", "Finished sending the messages", qos=1)
                p = multiprocessing.Process(
                    target=worker, args=[ret_value])
                p.start()

            if ret_value.value == 1:
                print("Timeout finished")
                avg_weights = average_model(
                    local_models_weight_list)
                model.set_weights(avg_weights)
                print("Testing global model")
                model_testing(model)
                publish_flag = True

        else:
            p = 2
