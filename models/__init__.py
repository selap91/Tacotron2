# code based on https://github.com/keithito/tacotron/

from .tacotron2 import Tacotron2


def create_model(name, hparas):
  if name == 'tacotron2':
    return Tacotron2(hparas)
  else:
    raise Exception('Unknown model: ' + name)
