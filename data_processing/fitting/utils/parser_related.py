
def str2bool(v):
  #susendberg's function
  return v.lower() in ("yes", "true", "t", "1")

def get_bool_type(parser):
    parser.register('type','bool',str2bool)