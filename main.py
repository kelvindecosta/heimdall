from lib.dataset import download, preprocess

if __name__ == "__main__":
    # Downloads and extracts the dataset idemptotently
    download()

    # Preprocesses the dataset by creating tiles for training
    preprocess()
