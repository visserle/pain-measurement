import sys

def flatten_list(l):
    """Flatten a list that is one level deep (i.e., a list of lists)."""
    return [item for sublist in l for item in sublist]

def print_size(obj):
    """Print the size of a DataFrame in MB."""
    print('obj size:', round(sys.getsizeof(obj) / 1024**2), 'MB')
