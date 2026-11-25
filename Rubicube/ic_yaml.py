import os
import json
import yaml

from params import p



def set_new_alert(ic_alert, yaml_dict):
    '''
    Used to re-write the IC job file with a new ic_alert. 
    
    :param ic_alert: dict, json schema object of the IC alert from the gcn kafka stream
    :param yaml_dict: dict, yaml parameters extractd from the yaml template
    '''
    
    map_name = ic_alert['healpix_url'].split('/')[-1]
    date_str = ic_alert['alert_datetime'].split('T')[0]
    
    #set new values in T4 Healpix Token Generator
    params = yaml_dict['task'][0]['config']['execute'][0]['config']['execute'][0]['config']
    
    params['map_name'] = map_name
    params['map_url'] = ic_alert['healpix_url']
    params['date_str'] = date_str
    params['map_dir'] = '/Users/cristozilleruelo/Ampel-HU-astro/RubiCube/ic_alerts_temp'
    
    #set new values
    yaml_dict['task'][0]['config']['execute'][0]['config']['execute'][0]['config'] = params

    #set the stream for the alert processor
    stream_list = yaml_dict['task'][1]['config']['supplier']['config']['loader']['config']['stream'].split('_')
    #str_header, name = stream_list[0].split('-')# (%%IceCube, 260115A)
    old_name = stream_list.pop(0) #removed
    new_name = '%%' + map_name.split('_')[0]
    stream_list.insert(0, new_name)

    #Set new value
    yaml_dict['task'][1]['config']['supplier']['config']['loader']['config']['stream'] = '_'.join(stream_list)


    
    #set new saving name in t3
    if type(ic_alert['event_name']) == list:
        ic_alert['event_name'] = ic_alert['event_name'][0]
    file_name = ic_alert['event_name'] + '.pdf'
    pdf_path = yaml_dict['task'][3]['config']['stage']['config']['execute'][0]['config']['pdf_path']
    directory = os.path.join(*pdf_path.split('/')[:-1])
    
    #set new path 
    yaml_dict['task'][3]['config']['stage']['config']['execute'][0]['config']['pdf_path'] = os.path.join(os.sep, directory, file_name)
    

    return yaml_dict

def write_new_yaml(ic_alert, yaml_name, save_name='', save_dir=p['file_dir']):
    '''
    Writes an amnpel job file for a new IceCube alert based on a provided job file,. 

    Params:
    -------
    ic_alert: dict or str, json schema of the IC alert from the gcn kafka stream
    yaml_path: str, name of the old ampel job file
    save_path: str, filename to be saved as. If '', then the yalm_path file gets re-written. 
    save_dir: str, path to directory where the yaml files are stored

    '''

    with open(os.path.join(save_dir, yaml_name), 'r') as file:
        yaml_dict = yaml.safe_load(file)
    
    if type(ic_alert) == str:
        #then the input is just the full alert string
        ic_alert = json.loads(ic_alert)
    
    new_yaml = set_new_alert(ic_alert, yaml_dict)

    if save_name == '':
        save_name = yaml_name
    with open(os.path.join(save_dir, save_name), 'w') as file:
        yaml.dump(new_yaml, file, sort_keys=False) #if sort_keys=True then file is written alphabetically
    
    return os.path.join(save_dir, save_name)
