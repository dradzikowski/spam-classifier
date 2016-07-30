import logging
import time

import matplotlib.pyplot as plt


def log_start_time():
    logging.info("--------- Start time: " + time.strftime("%Y-%m-%d %H:%M:%S"))
    start_time = time.time()
    return start_time


def log_end_time(start_time):
    logging.info("--- Execution took: %s seconds ---" % (time.time() - start_time))
    logging.info("--------- End time: " + time.strftime("%Y-%m-%d %H:%M:%S"))


def log_misclassified_emails(predictions, test_features, test_labels):
    j = 0
    for i in range(len(predictions)):
        if predictions[i] != test_labels[i]:
            j += 1
            logging.info("###################################################################################")
            logging.info("--- (((" + str(j) + "))) --- was: " + test_labels[i] + ", predicted: " + predictions[i])
            logging.info(test_features[i])
            logging.info("###################################################################################")


def log_results(confusion, data, scores):  # , confusion_matrix):
    print('--------------------------------------')
    # print classifier and settings
    print('Classified: ' + str(len(data)))
    print('Accuracy: ' + str(sum(scores) / len(scores)))
    print('Confusion matrix: \n' + str(confusion))
    print_plot(confusion)
    print('--------------------------------------')


def print_plot(confusion):
    plt.matshow(confusion)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def log_classification_parameters(pipeline):
    logging.info("--------------------")
    logging.info("Pipeline steps: ")
    for step in pipeline.steps:
        logging.info("--- Step: " + step.__str__())
    logging.info("--------------------")
