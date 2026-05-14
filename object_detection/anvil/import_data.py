import os
from roboflow import Roboflow

def main():
    # Get your cluster scratch directory
    scratch_dir = os.getenv('CLUSTER_SCRATCH')
    scratch_dir += '/projects'

    # PASTE ROBOFLOW DOWNLOAD LINK BELOW and make sure to delete the first and last line
    # (otherwise you'd download to the wrong place!)
                    
    dataset = version.download(model_format="yolov5", location=scratch_dir)
    get_yaml(dataset.location)
    
def get_yaml(path):
    with open(f'{path}/data.yaml', 'r') as f:
        with open(f'data.yaml', 'w') as df:
            lines = f.read().splitlines()
            for line in lines:
                if "test:" in line:
                    df.write(f"test: {path}/test/images")
                elif "train:" in line:
                    df.write(f"train: {path}/train/images")
                elif "val:" in line:
                    df.write(f"val: {path}/valid/images")
                else:
                    df.write(line)
                df.write('\n')

if __name__ == "__main__":
    main()