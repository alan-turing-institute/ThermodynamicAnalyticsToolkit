def get_list_from_string(str_or_list_of_str):
    """ Extracts list of ints from any string (or list of strings).

    :param str_or_list_of_str: string
    :return: list of ints
    """
    tmpstr=str_or_list_of_str
    if str_or_list_of_str is not str:
        try:
            tmpstr=" ".join(str_or_list_of_str)
        except(TypeError):
            tmpstr=" ".join([item for sublist in str_or_list_of_str for item in sublist])
    return [int(item) for item in tmpstr.split()]
