import numpy as np
import random
import time

from data_augmentation.utils.trim_dataset import TrimDataset


def generate_session(dataset: TrimDataset):
    """Return sessions and sessions_time"""
    print("Session generation started.", flush=True)
    start_time = time.time()
    sessions = dict()
    sessions_time = dict()
    for user_uid in dataset.inter_feat[dataset.uid_field].unique():
        user_id = str(user_uid)
        if user_id not in sessions:
            sessions[user_id] = []
            sessions_time[user_id] = []

        for inter in dataset.inter_feat[
            dataset.inter_feat[dataset.uid_field] == user_uid
        ].itertuples():
            sessions[user_id].append(str(getattr(inter, dataset.iid_field)))
            sessions_time[user_id].append(int(getattr(inter, dataset.time_field)))
    for user in sessions:
        # sort by sessions_time[user] in ascending order
        sessions[user] = [
            x for _, x in sorted(zip(sessions_time[user], sessions[user]))
        ]
        sessions_time[user] = sorted(sessions_time[user])
    end_time = time.time()
    print(
        f"Session generation finished, elapsed: {round(end_time-start_time, 4)} s",
        flush=True,
    )
    return sessions, sessions_time


def extract_train_valid_test(sessions, sessions_time):
    print("Dataset splitting started.", flush=True)
    start_time = time.time()
    train = dict()
    valid = dict()
    test = dict()
    train_time = dict()
    valid_time = dict()
    test_time = dict()
    for user in sessions:
        if len(sessions[user]) < 4:
            continue
        train[user] = sessions[user][:-2]
        train_time[user] = sessions_time[user][:-2]
        valid[user] = sessions[user][:-1]
        valid_time[user] = sessions_time[user][:-1]
        test[user] = sessions[user]
        test_time[user] = sessions_time[user]
    end_time = time.time()
    print(
        f"Dataset splitting finished, elapsed: {round(end_time - start_time, 4)} s",
        flush=True,
    )
    return train, valid, test, train_time, valid_time, test_time


def extract_cold_start_trivial(sessions, sessions_time, cold_start_ratio):
    """
    all_item: if True, all items in cold start users' sessions will be included
    only_target: if True, only valid and test target items in cold start users' sessions will be included. (only for all_item=True)
    """
    print(
        f"Extracting dataset for cold start, cold start ratio: {cold_start_ratio}",
        flush=True,
    )
    start_time = time.time()

    cold_start = dict()
    cold_start_time = dict()
    users = list(sessions.keys())
    users_num = len(users)
    cold_start_num = int(users_num * cold_start_ratio)
    np.random.seed(42)
    cold_start_user = np.random.choice(users, size=cold_start_num, replace=False)
    for remap_id in range(len(cold_start_user)):
        cold_start[str(remap_id)] = sessions[cold_start_user[remap_id]]
        cold_start_time[str(remap_id)] = sessions_time[cold_start_user[remap_id]]

    end_time = time.time()
    print(
        f"Extraction finished, elapsed: {round(end_time - start_time, 4)} s", flush=True
    )
    return cold_start, cold_start_time


def extract_cold_start(
    sessions, sessions_time, cold_start_ratio, all_item, only_target
):
    """
    all_item: if True, all items in cold start users' sessions will be included
    only_target: if True, only valid and test target items in cold start users' sessions will be included. (only for all_item=True)
    """
    print("Extracting dataset for cold start.", flush=True)
    start_time = time.time()
    cold_start = dict()
    cold_start_time = dict()
    users = list(sessions.keys())
    users_num = len(users)
    random.shuffle(users)
    cold_start_num = int(users_num * cold_start_ratio)
    if not all_item:
        for i in range(cold_start_num):
            if len(sessions[users[i]]) < 4:
                continue
            cold_start[users[i]] = sessions[users[i]][:-2]
            cold_start_time[users[i]] = sessions_time[users[i]][:-2]
        if len(cold_start) < cold_start_num:
            for i in range(cold_start_num, users_num):
                if len(sessions[users[i]]) < 4:
                    continue
                cold_start[users[i]] = sessions[users[i]][:-2]
                cold_start_time[users[i]] = sessions_time[users[i]][:-2]
                if len(cold_start) == cold_start_num:
                    break
    else:
        items_need = list()
        item_user = dict()  # item-user mapping
        item_cnt = dict()  # items count
        for k, v in sessions.items():
            if only_target:
                items_need += v[-2:]
            else:
                items_need += v
            for item in set(v):
                if item not in item_user:
                    item_user[item] = []
                    item_cnt[item] = 0
                item_user[item].append(k)
        items_need = set(items_need)
        existed_item = set()
        existed_user = set()
        for item in items_need:
            if item in existed_item:
                continue
            random.shuffle(item_user[item])
            for u in item_user[item]:
                if u in existed_user:
                    continue
                if len(sessions[u]) < 4:
                    continue
                cold_start[u] = sessions[u][:-2]
                cold_start_time[u] = sessions_time[u][:-2]
                existed_item = existed_item.union(set(cold_start[u]))
                existed_user.add(u)
                for i in cold_start[u]:
                    item_cnt[i] += 1
        duplicate_user = set()
        for k, v in cold_start.items():  # delete users who have no unique items
            has_unique = False
            for i in v:
                if item_cnt[i] == 1 and i in items_need:
                    has_unique = True
                    break
            if not has_unique:
                duplicate_user.add(k)
                existed_user.remove(k)
        for k in duplicate_user:
            cold_start.pop(k)
            cold_start_time.pop(k)

        if len(cold_start) < cold_start_num:  # add more users if not enough
            for i in range(cold_start_num, users_num):
                if users[i] in existed_user:
                    continue
                if len(sessions[users[i]]) < 4:
                    continue
                cold_start[users[i]] = sessions[users[i]][:-2]
                cold_start_time[users[i]] = sessions_time[users[i]][:-2]
                if len(cold_start) == cold_start_num:
                    break
    end_time = time.time()
    print(
        f"Extraction finished, elapsed: {round(end_time - start_time, 4)} s", flush=True
    )
    return cold_start, cold_start_time


def export_file(output_file_name, sessions, sessions_time):
    session_id = 1
    with open(output_file_name, "w") as f:
        f.write("session_id:token\titem_id_list:token_seq\titem_id:token\n")
        for i, user in enumerate(sessions):
            f.write(
                str(session_id)
                + "\t"
                + " ".join(sessions[user][:-1])
                + "\t"
                + sessions[user][-1]
                + "\n"
            )
            session_id += 1
    session_id = 1
    output_file_time_name = output_file_name.replace(".inter", ".time")
    with open(output_file_time_name, "w") as f:
        f.write("session_id:token\tsource_timestamp\ttarget_timestamp\n")
        for i, user in enumerate(sessions):
            f.write(
                str(session_id)
                + "\t"
                + " ".join([str(t) for t in sessions_time[user][:-1]])
                + "\t"
                + str(sessions_time[user][-1])
                + "\n"
            )
            session_id += 1
