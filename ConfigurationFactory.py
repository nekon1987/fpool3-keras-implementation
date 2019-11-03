class ConfigurationFactory(object):

    @staticmethod
    def CreateConfiguration():
        configuration = Configuration()
        configuration.training_loss_function = 'binary_crossentropy'
        configuration.number_of_epochs = 700
        configuration.early_stop_if_loss_functions_does_not_improve_limit = 150
        configuration.use_normalization = True

        configuration.notes_regarding_training_session = ''

        return configuration

class Configuration(object):

    def __init__(self):
        self.training_loss_function = 'n/a'
        self.number_of_epochs = -1
        self.early_stop_if_loss_functions_does_not_improve_limit = -1
        self.notes_regarding_training_session = 'n/a'
        self.use_normalization = True