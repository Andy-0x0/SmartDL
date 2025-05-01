import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class SlidingWindowSampler:

    @staticmethod
    def get_data_ser(data, width, offset, stride=1, padding_str=0, padding_end=0, padding_item=0):
        '''
        Get the sliding window data from the given time series (pd.Series).

        :param data:            the input data with type pd.Series
        :param width:           the width representing the sequence length of the data to be considered in one step
        :param offset:          the index of the sliding window's starting point
        :param stride:          the stride of each move
        :param padding_str:     the padding width before the data begin
        :param padding_end:     the padding width after the data begin
        :param padding_item:    the value to be put as the padding
        :return:                the sliding window data with type pd.Series
        '''
        if not isinstance(data, pd.Series):
            try:
                data = pd.Series(data)
            except TypeError:
                raise TypeError('[Sampler] Cannot convert input data to pd.Series')

        result_ser = data
        if padding_str:
            if padding_item is None:
                padding_item = data.iloc[0]
            result_ser = pd.concat([pd.Series([padding_item] * padding_str), result_ser], ignore_index=True)
        if padding_end:
            if padding_item is None:
                padding_item = data.iloc[-1]
            result_ser = pd.concat([result_ser, pd.Series([padding_item] * padding_end)], ignore_index=True)

        collector = []
        for idx in range(offset, len(result_ser) - width + 1, stride):
            if width <= 1:
                collector.append(result_ser.iloc[idx])
            else:
                collector.append(result_ser.iloc[idx: idx + width].to_numpy())

        return pd.Series(collector, name=data.name)

    @staticmethod
    def get_data_df(
        data: pd.DataFrame,
        on: tuple[int | str] | None,
        width: tuple[int] | int,
        offset: tuple[int] | int,
        stride: int = 1,
        padding_str: int = 0,
        padding_end: int = 0,
        padding_item: object = 0
    ) -> pd.DataFrame:
        '''
        Get the sliding window data from the given time series (pd.DataFrame).

        :param data:            the input data with type pd.DataFrame
        :param on:              the columns to be considered
        :param width:           the width representing the sequence length of the data to be considered in one step
        :param offset:          the index of the sliding window's starting point
        :param stride:          the stride of each move
        :param padding_str:     the padding width before the data begin
        :param padding_end:     the padding width after the data begin
        :param padding_item:    the value to be put as the padding
        :return:                the sliding window data with type pd.DataFrame
        '''
        # Transform the input 'data' into pd.Dataframe
        if not isinstance(data, pd.DataFrame):
            try:
                data = pd.DataFrame(data)
            except TypeError:
                raise TypeError('[Sampler] Cannot convert input data to pd.Dataframe')

        # Interpret the None meaning for param 'on'
        if on is None:
            on = list(range(len(data.columns)))

        # Broadcast the param 'width', 'offset' and 'padding_item' if received non-iterable types
        column_num = len(on)
        row_num = len(data)
        if not isinstance(width, (list, tuple, np.ndarray, pd.Series)):
            width = [width] * column_num
        if not isinstance(offset, (list, tuple, np.ndarray, pd.Series)):
            offset = [offset] * column_num
        if not isinstance(padding_item, (list, tuple, np.ndarray, pd.Series)):
            padding_item = [padding_item] * column_num

        # Generate the corresponding column names for returning pd.Dataframe
        def generate_columns(list_int, columns):
            list_str = list(map(lambda x: columns[x], list_int))

            map_dict = {}
            for name in list_str:
                map_dict[name] = map_dict.get(name, 0) + 1

            copy_dict = map_dict.copy()
            for idx, name in enumerate(list_str):
                map_dict[name] -= 1

                if copy_dict[name] - map_dict[name] < 2:
                    pass
                else:
                    list_str[idx] = str(list_str[idx]) + f"_{copy_dict[name] - map_dict[name] - 1}"

            return list_str
        on_int = list(map(lambda x: x if isinstance(x, int) else data.columns.get_loc(x), on))
        on_str = generate_columns(on_int, data.columns)

        # Calculate the end point for each column when doing the sliding window ensuring the alignment
        spaces = [a + b for a, b in zip(width, offset)]
        gaps = [max(spaces) - x for x in spaces]
        backs = list(map(lambda x: padding_end - x, gaps))

        # Generate each column as type (pd.Series) for the sliding window data representation
        collector = []
        for idx, col in enumerate(on_int):
            ser = SlidingWindowSampler.get_data_ser(data=data.iloc[:row_num + backs[idx], col],
                                                    width=width[idx],
                                                    offset=offset[idx],
                                                    stride=stride,
                                                    padding_str=padding_str,
                                                    padding_end=max(0, backs[idx]),
                                                    padding_item=padding_item[idx])
            collector.append(ser)

        # Generate the return pd.Dataframe
        return_df = pd.DataFrame({key: collector[idx] for idx, key in enumerate(on_str)})
        return return_df

    @staticmethod
    def get_data_merged(
        data: pd.DataFrame,
        packs: tuple[int | str | tuple[int] | tuple[str]],
        width: tuple[int] | int,
        offset: tuple[int | tuple[int]] | tuple[int] | int,
        stride: int = 1,
        padding_str: int = 0,
        padding_end: int = 0,
        padding_item: object = 0
    ) -> tuple[pd.Series]:
        '''
        Get the sliding window data from the given time series (pd.DataFrame), return the specified packs merged (pd.DataFrame)

        :param data:            the input data with type pd.DataFrame
        :param packs:           the packs of columns to be stacked as shape:[seq, *here*, ...]
        :param width:           the width representing the sequence length of the data to be considered in one step (# of ints == # of packs)
        :param offset:          the index of the sliding window's starting point (# of tuples == # of packs)
        :param stride:          the stride of each move
        :param padding_str:     the padding width before the data begin
        :param padding_end:     the padding width after the data begin
        :param padding_item:    the value to be put as the padding
        :return:                the sliding window data for each requested packs (pd.Series)
        '''
        # formate the params 'packs', 'width', 'offset', 'padding_item'
        def formate_packs(raw_packs, columns):
            fmt_packs = []
            for pack in raw_packs:
                if isinstance(pack, (list, tuple, np.ndarray, pd.Series)):
                    fmt_packs.append(tuple(map(lambda x: columns.get_loc(x) if isinstance(x, str) else x, pack)))
                elif isinstance(pack, str):
                    fmt_packs.append(tuple(columns.get_loc(pack)))
                else:
                    fmt_packs.append((pack, ))

            return fmt_packs

        def aline_accordance(src_list, tgt_list):
            global_broadcasting = True if not isinstance(tgt_list, (list, tuple, np.ndarray, pd.Series)) else False

            fmt_list = []
            for idx, src in enumerate(src_list):
                current_len = len(src)
                if global_broadcasting:
                    fmt_list.append(tuple([tgt_list] * current_len))
                elif not isinstance(tgt_list[idx], (list, tuple, np.ndarray, pd.Series)):
                    fmt_list.append(tuple([tgt_list[idx]] * current_len))
                elif isinstance(tgt_list[idx], (list, tuple, np.ndarray, pd.Series)) and len(tgt_list[idx]) == 1:
                    fmt_list.append(tuple([tgt_list[idx]] * current_len))
                else:
                    fmt_list.append(tgt_list[idx])

            return fmt_list

        packs_int = formate_packs(packs, data.columns)
        width = aline_accordance(packs_int, width)
        offset = aline_accordance(packs_int, offset)
        if padding_str or padding_end:
            padding_item = aline_accordance(packs_int, padding_item)

        # Generate each merged columns as type (pd.Series) for the sliding window data representation
        def unzip_packs(raw_packs):
            unzip_list = []
            for pack in raw_packs:
                if isinstance(pack, (tuple, list, np.ndarray, pd.Series)):
                    unzip_list.extend(list(pack))
                else:
                    unzip_list.append((pack, ))

            return unzip_list

        tot_columns = unzip_packs(packs_int)
        tot_width = unzip_packs(width)
        tot_offset = unzip_packs(offset)
        if padding_str or padding_end:
            tot_padding_it = unzip_packs(padding_item)
        else:
            tot_padding_it = None

        source_df = SlidingWindowSampler.get_data_df(data,
                                                     on=tot_columns,
                                                     width=tot_width,
                                                     offset=tot_offset,
                                                     stride=stride,
                                                     padding_str=padding_str,
                                                     padding_end=padding_end,
                                                     padding_item=tot_padding_it)

        return_sers = []
        last_end = 0
        prefix = r"pack_"
        for idx in range(len(packs_int)):
            template_df = source_df.iloc[:, last_end: last_end + len(packs_int[idx])]

            if len(template_df.columns) > 1:
                result_ser = template_df.apply(lambda row: np.stack(row.values, axis=1), axis=1)
            else:
                result_ser = template_df.squeeze()

            return_sers.append(pd.Series(result_ser, name=prefix + str(idx + 1)))
            last_end += len(packs_int[idx])

        return tuple(return_sers)


class TransformerSampler:
    def __init__(self, data):
        self.data = pd.DataFrame(data)

    def get_data_df(self, features, label, enc_width, dec_width, l_width, enc_offset, dec_offset, l_offset):
        def format_tuple(value):
            if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
                return tuple(value)
            else:
                return value

        enc_ser, dec_ser, l_ser = SlidingWindowSampler.get_data_merged(data=self.data,
                                                                       packs=(format_tuple(features), format_tuple(label), format_tuple(label)),
                                                                       width=(format_tuple(enc_width), format_tuple(dec_width), format_tuple(l_width)),
                                                                       offset=(format_tuple(enc_offset), format_tuple(dec_offset), format_tuple(l_offset)))
        return enc_ser, dec_ser, l_ser

    def get_data_loader(self, features, label, enc_width, dec_width, l_width, enc_offset, dec_offset, l_offset, batch_size=64, shuffle=True):
        enc_ser, dec_ser, l_ser = self.get_data_df(features, label, enc_width, dec_width, l_width, enc_offset, dec_offset, l_offset)

        class ReturnDataset(Dataset):
            def __init__(self, source_enc_ser, source_dec_ser, source_l_ser):
                self.source_enc_ser = source_enc_ser
                self.source_dec_ser = source_dec_ser
                self.source_l_ser = source_l_ser

            def __len__(self):
                len_enc = len(self.source_enc_ser)
                len_dec = len(self.source_dec_ser)
                len_l = len(self.source_l_ser)

                if len_enc == len_dec and len_dec == len_l:
                    return len_enc
                else:
                    raise ValueError('[Sampler] The length of the encoder, decoder and label series are not identical')

            def __getitem__(self, index):
                enc_item = torch.tensor(self.source_enc_ser.iloc[index].astype(np.float32))
                dec_item = torch.tensor(self.source_dec_ser.iloc[index].astype(np.int64))
                l_item = torch.tensor(self.source_l_ser.iloc[index].astype(np.int64))

                # Activate the following line only if you are doing regression
                # while dec_item.dim() < 3 - 1:
                #     dec_item = dec_item.unsqueeze(-1)

                # l_item = l_item.unsqueeze(-1)

                return (enc_item, dec_item), l_item

        return_dataloader = DataLoader(dataset=ReturnDataset(enc_ser, dec_ser, l_ser),
                                       batch_size=batch_size,
                                       shuffle=shuffle)

        return return_dataloader








