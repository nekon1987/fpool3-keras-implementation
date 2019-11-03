import DatasetFactory, ModelFactory
from ModelService import ModelService

class main:

    @staticmethod
    def execute_training_and_evaluation():
        training_data_x, training_labels_y = DatasetFactory.load_training_data_and_its_labels()
        evaluation_data_x, evaluation_labels_y = DatasetFactory.load_evaluation_data_and_its_labels()

        network_model = ModelFactory.create_fpool3_model()

        ModelService.compile_and_fit_the_model(network_model, training_data_x, training_labels_y)
        accuracy = ModelService.evaluate_the_model(network_model, evaluation_data_x, evaluation_labels_y)

        file_name = ModelService.save_model_and_configuration(network_model, accuracy)

        print('Jarvis hot word detector training has completed. File stored @ ' + file_name)

main.execute_training_and_evaluation()