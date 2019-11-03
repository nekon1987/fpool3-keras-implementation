import keras, datetime, os, yaml, sys
import numpy as np
from ConfigurationFactory import ConfigurationFactory

class ModelService(object):

    @staticmethod
    def compile_and_fit_the_model(model, train_data_x, train_labels_y):
        configuration = ConfigurationFactory.CreateConfiguration()

        model.compile(
            loss=configuration.training_loss_function,
            optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)  ,
            metrics=['accuracy'])

        early_stopping = keras.callbacks.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                         patience=configuration.early_stop_if_loss_functions_does_not_improve_limit,
                         verbose=1, mode='min', baseline=None, restore_best_weights=True)

        history = model.fit(
            train_data_x, train_labels_y,
            callbacks=[early_stopping],
            validation_split=0.2,
            epochs=configuration.number_of_epochs,
            verbose=1,
            batch_size=670)

        return history

    @staticmethod
    def evaluate_the_model(network_model, evaluation_data_x, evaluation_labels_y):
        score = network_model.evaluate(evaluation_data_x, evaluation_labels_y, verbose=1)  # different set !!!

        print('Evaluation results:')
        print( network_model.metrics_names[0] + ' -> ' + str(score[0]))
        print( network_model.metrics_names[1] + ' -> ' + str(score[1]))

        for index in list(range(30)):
            test = evaluation_data_x[index]
            test = np.reshape(test, [1, 40, 32, 1])
            res = network_model.predict(test, verbose=1)
            print('label -> ' + str(evaluation_labels_y[index]) + ' | ' + str(res))

        return score[1]

    @staticmethod
    def save_model_and_configuration(network_model, evaluation_accuracy):
        configuration = ConfigurationFactory.CreateConfiguration()

        test_descriptive_label = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + '-' + 'jarvis-hot-word-detector-ACC-' + str(evaluation_accuracy)

        results_directory = os.getcwd() + '\\' + 'output_model\\';
        os.mkdir(results_directory + test_descriptive_label)
        model_filename = results_directory + test_descriptive_label + '\\' + test_descriptive_label + '.h5'

        with open(results_directory + test_descriptive_label + '\\' + 'configuration.json', 'w') as f:
            yaml.dump(configuration, f)

        network_model.save(model_filename)

        return test_descriptive_label + '.h5'
