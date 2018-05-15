AUs = [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24]
AUs_name_en = ['Inner Brow Raiser\n', \
      'Outer Brow Raiser\n', \
      'Brow Lowerer\n', \
      'Cheek Raiser\n', \
      'Lid Tightener\n', \
      'Upper Lip Raiser\n', \
      'Lip Corner Puller\n', \
      'Dimpler\n', \
      'Lip Corner Depressor\n', \
      'Chin Raiser\n', \
      'Lip Tightener\n', \
      'Lip Pressor\n', \
      ]

AUs_name_es= ['Ceja Interior Levantada\n', \
      'Ceja Exterior Levantada\n', \
      'Cejas fruncidas\n', \
      'Mejillas levantadas\n', \
      'Parpados apretados\n', \
      'Labio Superior Levantado\n', \
      'Esquina Labial Estirada\n', \
      'Hoyuelo Facial\n', \
      'Esquina Labial hacia abajo\n', \
      'Barbilla levantada\n', \
      'Labios mordidos\n', \
      'Labios apretados\n', \
      ]

TXT_PATH='/home/afromero/datos2/AUNets/data'

def update_folder(config, folder):
  import os
  config.log_path = os.path.join(config.log_path, folder)
  config.model_save_path = os.path.join(config.model_save_path, folder)

def remove_folder(config):
  import os
  logs = config.log_path
  models = config.model_save_path
  print("YOU ARE ABOUT TO REMOVE EVERYTHING IN:\n{}\n{}".format(logs, models))
  print("YOU ARE ABOUT TO REMOVE EVERYTHING IN:\n{}\n{}".format(logs, models))
  raw_input("ARE YOU SURE?")
  os.system("rm -r {} {}".format(logs, models))

def update_config(config):
  import os, glob, math, imageio

  if config.SHOW_MODEL: config.batch_size=1

  config.OF_option = config.OF
  if config.OF!='None': 
    config.OF=True
  else: 
    config.OF=False     

  folder_parameters = os.path.join(config.dataset, config.mode_data, 'fold_'+config.fold, 'AU'+str(config.AU).zfill(2))
  update_folder(config, folder_parameters)
  config.metadata_path = os.path.join(config.metadata_path, folder_parameters)
  if config.HYDRA: update_folder(config, 'HYDRA')
  update_folder(config, 'OF_'+config.OF_option)
  update_folder(config, config.finetuning)

  if config.DELETE:
    remove_folder(config)
  
  config.xlsfile = os.path.join(config.results_path, config.mode_data, config.finetuning+'.xlsx')

  if config.pretrained_model=='':
    try:
      # ipdb.set_trace()
      config.pretrained_model = sorted(glob.glob(os.path.join(config.model_save_path, '*.pth')))[-1]
      config.pretrained_model = os.path.basename(config.pretrained_model).split('.')[0]
    except:
      pass

  if config.test_model=='':
    try:
      # ipdb.set_trace()
      config.test_model = sorted(glob.glob(os.path.join(config.model_save_path, '*.pth')))[-1]
      config.test_model = os.path.basename(config.test_model).split('.')[0]
    except:
      config.test_model = ''  

  if not os.path.exists(os.path.dirname(config.xlsfile)):
    os.makedirs(os.path.dirname(config.xlsfile))


  return config