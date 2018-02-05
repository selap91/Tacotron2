from .tacotron2 import Tacotron2


def create_model(name, hparams):
  if name == 'Tacotron2':
    return Tacotron2(hparams)
  else:
    raise Exception('Unknown model: ' + name)
