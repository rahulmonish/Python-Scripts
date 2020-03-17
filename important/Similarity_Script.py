def similarity(df_list, acc):
    copy_list = copy.deepcopy(df_list)
    added_index = []
    count = 0

    for i in range(0, len(df_list)):
        if i not in added_index:

            index_list = []
            index_list.append(i)
            for j in range(i + 1, len(df_list)):

                if j not in added_index:
                    ratio = SequenceMatcher(None, df_list[i], df_list[j]).ratio()
                    ratio *= 100

                    if ratio > acc:
                        index_list.append(j)
                        added_index.append(j)

            for index in index_list:
                copy_list[index] = count

        count += 1

    return copy_list