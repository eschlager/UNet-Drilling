import socket

# if you want to run the code on a different server/laptop, add the path
# to your local data here

paths = {'kcs-cuda.hadoop.know-center.at': '/home/eschlager/SmartDrilling/',
         'kcnb-eschlager': '/Users/eschlager/Documents/SmartDrilling/'}


def get_base_path_for_current_host():
    hostname = socket.gethostname()

    if hostname not in paths:
        raise ValueError(f'Your hostname {hostname} is not in the list!')
    return paths[hostname]
