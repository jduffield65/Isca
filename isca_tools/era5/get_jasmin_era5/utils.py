import numpy as np
import xarray as xr

def convert_lnsp_to_sp(ds: xr.Dataset, delete_lnsp: bool = True) -> xr.Dataset:
    """
    Function to convert logarithm of surface pressure, into surface pressure with units of Pa

    Args:
        ds: Dataset containing variable called `lnsp` to be converted.
        delete_lnsp: If True, delete the original `lnsp` variable.

    Returns:
        Dataset containing variable called `sp`
    """
    if 'lnsp' not in ds:
        print("Dataset does not contain variable called 'lnsp', returning original dataset.")
        return ds

    # Compute surface pressure
    sp = np.exp(ds['lnsp'])

    # Rename and set attributes
    sp.name = 'sp'
    sp.attrs['long_name'] = 'surface pressure'
    sp.attrs['units'] = 'Pa'

    # Drop lnsp and add sp
    if delete_lnsp:
        ds = ds.drop_vars('lnsp')
    ds['sp'] = sp

    return ds



def get_ab(n_levels):
    """
    --------------------------------------------------------------------------------
    Returns ECMWF hybrid pressure level a and b coefficients for the specified
    level definition.

    Arguments:
        int :: n_levels
            Number of levels to provide coefficients for. Must be one of the
            following ECMWF level definitions:
                137, 91, 62, 60, 50, 40, 31, 19, 16

    Returns:
        numpy array :: a (length = n_levels + 1)
            Floating point a coefficients in Pa
        numpy array :: b (length = n_levels + 1)
            Floating point b coefficients

    --------------------------------------------------------------------------------
    """
    level_list = [137, 91, 62, 60, 50, 40, 31, 19, 16]
    if n_levels not in level_list:
        raise Exception(
            """Get_ab: Number of levels input not recognised as a
                        valid ECMWF level definition. n_levels must be one of
                        the following: """
            + str(level_list)
        )
    if n_levels == 137:
        a = [
            0.0,
            2.000365,
            3.102241,
            4.666084,
            6.827977,
            9.746966,
            13.605424,
            18.608931,
            24.985718,
            32.98571,
            42.879242,
            54.955463,
            69.520576,
            86.895882,
            107.415741,
            131.425507,
            159.279404,
            191.338562,
            227.968948,
            269.539581,
            316.420746,
            368.982361,
            427.592499,
            492.616028,
            564.413452,
            643.339905,
            729.744141,
            823.967834,
            926.34491,
            1037.201172,
            1156.853638,
            1285.610352,
            1423.770142,
            1571.622925,
            1729.448975,
            1897.519287,
            2076.095947,
            2265.431641,
            2465.770508,
            2677.348145,
            2900.391357,
            3135.119385,
            3381.743652,
            3640.468262,
            3911.490479,
            4194.930664,
            4490.817383,
            4799.149414,
            5119.89502,
            5452.990723,
            5798.344727,
            6156.074219,
            6526.946777,
            6911.870605,
            7311.869141,
            7727.412109,
            8159.354004,
            8608.525391,
            9076.400391,
            9562.682617,
            10065.97852,
            10584.63184,
            11116.66211,
            11660.06738,
            12211.54785,
            12766.87305,
            13324.66895,
            13881.33106,
            14432.13965,
            14975.61523,
            15508.25684,
            16026.11523,
            16527.32227,
            17008.78906,
            17467.61328,
            17901.62109,
            18308.43359,
            18685.71875,
            19031.28906,
            19343.51172,
            19620.04297,
            19859.39063,
            20059.93164,
            20219.66406,
            20337.86328,
            20412.30859,
            20442.07813,
            20425.71875,
            20361.81641,
            20249.51172,
            20087.08594,
            19874.02539,
            19608.57227,
            19290.22656,
            18917.46094,
            18489.70703,
            18006.92578,
            17471.83984,
            16888.6875,
            16262.04688,
            15596.69531,
            14898.45313,
            14173.32422,
            13427.76953,
            12668.25781,
            11901.33984,
            11133.30469,
            10370.17578,
            9617.515625,
            8880.453125,
            8163.375,
            7470.34375,
            6804.421875,
            6168.53125,
            5564.382813,
            4993.796875,
            4457.375,
            3955.960938,
            3489.234375,
            3057.265625,
            2659.140625,
            2294.242188,
            1961.5,
            1659.476563,
            1387.546875,
            1143.25,
            926.507813,
            734.992188,
            568.0625,
            424.414063,
            302.476563,
            202.484375,
            122.101563,
            62.78125,
            22.835938,
            3.757813,
            0,
            0,
        ]
        b = [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0.000007,
            0.000024,
            0.000059,
            0.000112,
            0.000199,
            0.00034,
            0.000562,
            0.00089,
            0.001353,
            0.001992,
            0.002857,
            0.003971,
            0.005378,
            0.007133,
            0.009261,
            0.011806,
            0.014816,
            0.018318,
            0.022355,
            0.026964,
            0.032176,
            0.038026,
            0.044548,
            0.051773,
            0.059728,
            0.068448,
            0.077958,
            0.088286,
            0.099462,
            0.111505,
            0.124448,
            0.138313,
            0.153125,
            0.16891,
            0.185689,
            0.203491,
            0.222333,
            0.242244,
            0.263242,
            0.285354,
            0.308598,
            0.332939,
            0.358254,
            0.384363,
            0.411125,
            0.438391,
            0.466003,
            0.4938,
            0.521619,
            0.549301,
            0.576692,
            0.603648,
            0.630036,
            0.655736,
            0.680643,
            0.704669,
            0.727739,
            0.749797,
            0.770798,
            0.790717,
            0.809536,
            0.827256,
            0.843881,
            0.859432,
            0.873929,
            0.887408,
            0.8999,
            0.911448,
            0.922096,
            0.931881,
            0.94086,
            0.949064,
            0.95655,
            0.963352,
            0.969513,
            0.975078,
            0.980072,
            0.984542,
            0.9885,
            0.991984,
            0.995003,
            0.99763,
            1,
        ]
    elif n_levels == 60:
        a = [
            0.0,
            20.0,
            38.425343,
            63.647804,
            95.636963,
            134.483307,
            180.584351,
            234.779053,
            298.495789,
            373.971924,
            464.618134,
            575.651001,
            713.218079,
            883.660522,
            1094.834717,
            1356.474609,
            1680.640259,
            2082.273926,
            2579.888672,
            3196.421631,
            3960.291504,
            4906.708496,
            6018.019531,
            7306.631348,
            8765.053711,
            10376.12695,
            12077.44629,
            13775.3252,
            15379.80566,
            16819.47461,
            18045.18359,
            19027.69531,
            19755.10938,
            20222.20508,
            20429.86328,
            20384.48047,
            20097.40234,
            19584.33008,
            18864.75,
            17961.35742,
            16899.46875,
            15706.44727,
            14411.12402,
            13043.21875,
            11632.75879,
            10209.50098,
            8802.356445,
            7438.803223,
            6144.314941,
            4941.77832,
            3850.91333,
            2887.696533,
            2063.779785,
            1385.912598,
            855.361755,
            467.333588,
            210.39389,
            65.889244,
            7.367743,
            0.0,
            0.0,
        ]
        b = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.000076,
            0.000461,
            0.001815,
            0.005081,
            0.011143,
            0.020678,
            0.034121,
            0.05169,
            0.073534,
            0.099675,
            0.130023,
            0.164384,
            0.202476,
            0.243933,
            0.288323,
            0.335155,
            0.383892,
            0.433963,
            0.484772,
            0.53571,
            0.586168,
            0.635547,
            0.683269,
            0.728786,
            0.771597,
            0.811253,
            0.847375,
            0.879657,
            0.907884,
            0.93194,
            0.951822,
            0.967645,
            0.979663,
            0.98827,
            0.994019,
            0.99763,
            1.0,
        ]
    elif n_levels == 91:
        a = [
            2.00004,
            3.980832,
            7.387186,
            12.908319,
            21.413612,
            33.952858,
            51.746601,
            76.167656,
            108.715561,
            150.986023,
            204.637451,
            271.356506,
            352.824493,
            450.685791,
            566.519226,
            701.813354,
            857.945801,
            1036.166504,
            1237.585449,
            1463.16394,
            1713.709595,
            1989.87439,
            2292.155518,
            2620.898438,
            2976.302246,
            3358.425781,
            3767.196045,
            4202.416504,
            4663.776367,
            5150.859863,
            5663.15625,
            6199.839355,
            6759.727051,
            7341.469727,
            7942.92627,
            8564.624023,
            9208.305664,
            9873.560547,
            10558.88184,
            11262.48438,
            11982.66211,
            12713.89746,
            13453.22559,
            14192.00977,
            14922.68555,
            15638.05371,
            16329.56055,
            16990.62305,
            17613.28125,
            18191.0293,
            18716.96875,
            19184.54492,
            19587.51367,
            19919.79688,
            20175.39453,
            20348.91602,
            20434.1582,
            20426.21875,
            20319.01172,
            20107.03125,
            19785.35742,
            19348.77539,
            18798.82227,
            18141.29688,
            17385.5957,
            16544.58594,
            15633.56641,
            14665.64551,
            13653.21973,
            12608.38379,
            11543.16699,
            10471.31055,
            9405.222656,
            8356.25293,
            7335.164551,
            6353.920898,
            5422.802734,
            4550.21582,
            3743.464355,
            3010.146973,
            2356.202637,
            1784.854614,
            1297.656128,
            895.193542,
            576.314148,
            336.772369,
            162.043427,
            54.208336,
            6.575628,
            0.00316,
            0,
        ]
        b = [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0.000014,
            0.000055,
            0.000131,
            0.000279,
            0.000548,
            0.001,
            0.001701,
            0.002765,
            0.004267,
            0.006322,
            0.009035,
            0.012508,
            0.01686,
            0.022189,
            0.02861,
            0.036227,
            0.045146,
            0.055474,
            0.067316,
            0.080777,
            0.095964,
            0.112979,
            0.131935,
            0.152934,
            0.176091,
            0.20152,
            0.229315,
            0.259554,
            0.291993,
            0.326329,
            0.362203,
            0.399205,
            0.436906,
            0.475016,
            0.51328,
            0.551458,
            0.589317,
            0.626559,
            0.662934,
            0.698224,
            0.732224,
            0.764679,
            0.795385,
            0.824185,
            0.85095,
            0.875518,
            0.897767,
            0.917651,
            0.935157,
            0.950274,
            0.963007,
            0.973466,
            0.982238,
            0.989153,
            0.994204,
            0.99763,
            1,
        ]
    elif n_levels == 62:
        a = [
            0,
            988.835876,
            1977.67627,
            2966.516602,
            3955.356934,
            4944.197266,
            5933.037598,
            6921.870117,
            7909.441406,
            8890.707031,
            9860.52832,
            10807.7832,
            11722.74902,
            12595.00684,
            13419.46387,
            14192.00977,
            14922.68555,
            15638.05371,
            16329.56055,
            16990.62305,
            17613.28125,
            18191.0293,
            18716.96875,
            19184.54492,
            19587.51367,
            19919.79688,
            20175.39453,
            20348.91602,
            20434.1582,
            20426.21875,
            20319.01172,
            20107.03125,
            19785.35742,
            19348.77539,
            18798.82227,
            18141.29688,
            17385.5957,
            16544.58594,
            15633.56641,
            14665.64551,
            13653.21973,
            12608.38379,
            11543.16699,
            10471.31055,
            9405.222656,
            8356.25293,
            7335.164551,
            6353.920898,
            5422.802734,
            4550.21582,
            3743.464355,
            3010.146973,
            2356.202637,
            1784.854614,
            1297.656128,
            895.193542,
            576.314148,
            336.772369,
            162.043427,
            54.208336,
            6.575628,
            0.00316,
            0,
        ]
        b = [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0.000013,
            0.000087,
            0.000275,
            0.000685,
            0.001415,
            0.002565,
            0.004187,
            0.006322,
            0.009035,
            0.012508,
            0.01686,
            0.022189,
            0.02861,
            0.036227,
            0.045146,
            0.055474,
            0.067316,
            0.080777,
            0.095964,
            0.112979,
            0.131935,
            0.152934,
            0.176091,
            0.20152,
            0.229315,
            0.259554,
            0.291993,
            0.326329,
            0.362203,
            0.399205,
            0.436906,
            0.475016,
            0.51328,
            0.551458,
            0.589317,
            0.626559,
            0.662934,
            0.698224,
            0.732224,
            0.764679,
            0.795385,
            0.824185,
            0.85095,
            0.875518,
            0.897767,
            0.917651,
            0.935157,
            0.950274,
            0.963007,
            0.973466,
            0.982238,
            0.989153,
            0.994204,
            0.99763,
            1,
        ]
    elif n_levels == 50:
        a = [
            0,
            20.006149,
            43.29781,
            75.34623,
            115.082146,
            161.897491,
            215.896912,
            278.005798,
            350.138184,
            435.562286,
            539.651489,
            668.61554,
            828.398987,
            1026.366943,
            1271.644531,
            1575.537842,
            1952.054443,
            2418.549805,
            2996.526611,
            3712.626221,
            4599.856934,
            5699.114746,
            6998.388184,
            8507.411133,
            10181.70703,
            11883.08984,
            13442.91504,
            14736.35449,
            15689.20606,
            16266.60938,
            16465.00391,
            16297.62012,
            15791.59766,
            14985.26953,
            13925.51953,
            12665.29492,
            11261.23047,
            9771.40625,
            8253.210938,
            6761.339844,
            5345.917969,
            4050.71875,
            2911.570312,
            1954.804688,
            1195.890625,
            638.148438,
            271.625,
            72.0625,
            0,
            0,
            0,
        ]
        b = [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0.0001,
            0.000673,
            0.003163,
            0.009292,
            0.020319,
            0.036975,
            0.059488,
            0.087895,
            0.122004,
            0.161442,
            0.205703,
            0.254189,
            0.306235,
            0.361145,
            0.418202,
            0.476688,
            0.535887,
            0.595084,
            0.653565,
            0.710594,
            0.765405,
            0.817167,
            0.864956,
            0.907716,
            0.944213,
            0.972985,
            0.992281,
            1,
        ]
    elif n_levels == 40:
        a = [
            0,
            2000,
            4000,
            6000,
            8000,
            9988.882838,
            11914.52447,
            13722.94294,
            15369.73086,
            16819.47627,
            18045.18359,
            19027.69448,
            19755.10876,
            20222.20531,
            20429.86297,
            20384.48143,
            20097.40215,
            19584.32924,
            18864.75039,
            17961.35774,
            16899.46879,
            15706.44732,
            14411.12426,
            13043.21862,
            11632.75836,
            10209.50134,
            8802.356155,
            7438.803092,
            6144.315003,
            4941.778213,
            3850.913422,
            2887.696603,
            2063.779905,
            1385.912553,
            855.36175,
            467.333577,
            210.393894,
            65.889243,
            7.367743,
            0,
            0,
        ]
        b = [
            0,
            0,
            0,
            0,
            0,
            0.000197,
            0.001511,
            0.004884,
            0.011076,
            0.020678,
            0.034121,
            0.05169,
            0.073534,
            0.099675,
            0.130023,
            0.164384,
            0.202476,
            0.243933,
            0.288323,
            0.335155,
            0.383892,
            0.433963,
            0.484772,
            0.53571,
            0.586168,
            0.635547,
            0.683269,
            0.728786,
            0.771597,
            0.811253,
            0.847375,
            0.879657,
            0.907884,
            0.93194,
            0.951822,
            0.967645,
            0.979663,
            0.98827,
            0.994019,
            0.99763,
            1,
        ]
    elif n_levels == 31:
        a = [
            0,
            2000,
            4000,
            6000,
            8000,
            9976.135361,
            11820.53962,
            13431.39393,
            14736.35691,
            15689.20746,
            16266.6105,
            16465.00573,
            16297.61933,
            15791.5986,
            14985.26963,
            13925.51786,
            12665.29166,
            11261.22888,
            9771.40629,
            8253.212096,
            6761.341326,
            5345.91424,
            4050.717678,
            2911.569385,
            1954.805296,
            1195.889791,
            638.148911,
            271.626545,
            72.063577,
            0,
            0,
            0,
        ]
        b = [
            0,
            0,
            0,
            0,
            0,
            0.000391,
            0.00292,
            0.009194,
            0.020319,
            0.036975,
            0.059488,
            0.087895,
            0.122004,
            0.161442,
            0.205703,
            0.254189,
            0.306235,
            0.361145,
            0.418202,
            0.476688,
            0.535887,
            0.595084,
            0.653565,
            0.710594,
            0.765405,
            0.817167,
            0.864956,
            0.907716,
            0.944213,
            0.972985,
            0.992281,
            1,
        ]
    elif n_levels == 19:
        a = [
            2000,
            4000,
            6046.110595,
            8267.92756,
            10609.51323,
            12851.10017,
            14698.49809,
            15861.12518,
            16116.23661,
            15356.92412,
            13621.4604,
            11101.56199,
            8127.144155,
            5125.141747,
            2549.969411,
            783.195032,
            0,
            0,
            0,
        ]
        b = [
            0,
            0,
            0.000339,
            0.003357,
            0.01307,
            0.034077,
            0.07065,
            0.125917,
            0.201195,
            0.29552,
            0.405409,
            0.524932,
            0.646108,
            0.759698,
            0.856438,
            0.928747,
            0.972985,
            0.992281,
            1,
        ]
    elif n_levels == 16:
        a = [
            0,
            5000,
            9890.52,
            14166.3,
            17346.07,
            19121.15,
            19371.25,
            18164.47,
            15742.18,
            12488.05,
            8881.824,
            5437.539,
            2626.258,
            783.2966,
            0,
            0,
            0,
        ]
        b = [
            0,
            0,
            0.001721,
            0.013198,
            0.042217,
            0.093762,
            0.169571,
            0.268016,
            0.384274,
            0.510831,
            0.638268,
            0.756385,
            0.855613,
            0.928746,
            0.972985,
            0.992282,
            1,
        ]
    a = np.array(a).astype("float")
    b = np.array(b).astype("float")
    return a, b


def get_ph(ps, n_levels):
    """
    --------------------------------------------------------------------------------
    Returns pressure on half levels for ECMWF hybrid pressure levels.

    Arguments:
        numpy array or scalar :: ps
            Surface pressure in Pa
        int :: n_levels
            Number of levels to provide coefficients for. Must be one of the
            following ECMWF level definitions:
                137, 91, 62, 60, 50, 40, 31, 19, 16

    Returns:
        numpy array :: ph (shape = (n_levels + 1, ps.shape)
            Floating point array of pressure values on half levels in units Pa

    --------------------------------------------------------------------------------
    """
    a, b = get_ab(n_levels)
    try:
        ps_shape = ps.shape
    except:
        if hasattr(ps, "__iter__"):
            raise Exception(
                """get_ph: ps argument must be a numpy array or
                            scalar value"""
            )
        ps_shape = ()
    level_shape = np.ones(len(ps_shape) + 1).astype("int")
    level_shape[0] = n_levels + 1
    level_shape = tuple(level_shape)
    ph = a.reshape(level_shape) + b.reshape(level_shape) * ps
    return ph


def get_pl(ps, n_levels):
    """
    --------------------------------------------------------------------------------
    Returns pressure on full levels for ECMWF hybrid pressure levels.

    Arguments:
        numpy array or scalar :: ps
            Surface pressure in Pa
        int :: n_levels
            Number of levels to provide coefficients for. Must be one of the
            following ECMWF level definitions:
                137, 91, 62, 60, 50, 40, 31, 19, 16

    Returns:
        numpy array :: pl (shape = (n_levels, ps.shape)
            Floating point array of pressure on full levels in units Pa

    --------------------------------------------------------------------------------
    """
    ph = get_ph(ps, n_levels)
    pl = (ph[1:] + ph[:-1]) * 0.5
    return pl


def get_dp(ps, n_levels):
    """
    --------------------------------------------------------------------------------
    Returns difference in pressure between levels for ECMWF hybrid pressure
    levels.

    Arguments:
        numpy array or scalar :: ps
            Surface pressure in Pa
        int :: n_levels
            Number of levels to provide coefficients for. Must be one of the
            following ECMWF level definitions:
                137, 91, 62, 60, 50, 40, 31, 19, 16

    Returns:
        numpy array :: pl (shape = (n_levels, ps.shape)
            Floating point array of pressure differential between full levels in
            units Pa

    --------------------------------------------------------------------------------
    """
    ph = get_ph(ps, n_levels)
    dp = ph[1:] - ph[:-1]
    return dp


def get_gz(ps, gzs, T, q, n_levels):
    """
    --------------------------------------------------------------------------------
    Calculates the geopotential on model levels for ECMWF hybrid pressure
    levels.

    Arguments:
        numpy array or scalar :: ps
            Surface pressure in Pa
        numpy array or scalar :: gzs
            Surface geopotential in m**2 s**-2
        numpy array :: T
            Atmospheric temperature on levels in K. Must have shape (n_levels,
            ps.shape)
        numpy array :: q
            Atmospheric specific humidity on levels in kg kg**-1. Must have
            shape (n_levels, ps.shape)
        int :: n_levels
            Number of levels to provide coefficients for. Must be one of the
            following ECMWF level definitions:
                137, 91, 62, 60, 50, 40, 31, 19, 16

    Returns:
        numpy array :: gzf (shape = (n_levels, ps.shape)
            Floating point array of geopotential values on full levels in units
            m**2 s**-2

    --------------------------------------------------------------------------------
    """
    # Check that the dimensions of all inputs are consistent
    try:
        ps_shape = ps.shape
    except:
        if hasattr(ps, "__iter__"):
            raise Exception(
                """get_gz: ps argument must be a numpy array or
                            scalar value"""
            )
        ps_shape = ()
    try:
        gzs_shape = gzs.shape
    except:
        if hasattr(gzs, "__iter__"):
            raise Exception(
                """get_gz: gzs argument must be a numpy array or
                            scalar value"""
            )
        gzs_shape = ()
    if len(gzs_shape) == len(ps_shape):
        if gzs_shape != ps_shape:
            if len(gzs_shape) != 0 and len(ps_shape) != 0:
                raise Exception(
                    """get_gz: ps and gzs inputs must have the same
                                shape or have scalar values"""
                )
    elif len(gzs_shape) == (len(ps_shape) - 1):
        if gzs_shape != ps_shape[1:]:
            if len(gzs_shape) != 0 and len(ps_shape) != 0:
                raise Exception(
                    """get_gz: ps and gzs inputs must have the same
                                shape or have scalar values"""
                )
    else:
        raise Exception(
            """get_gz: ps and gzs inputs must have the same
                                shape or have scalar values"""
        )
    if len(ps_shape) != 0:
        out_shape = (n_levels,) + ps_shape
    elif len(gzs_shape) != 0:
        out_shape = (n_levels,) + gzs_shape
    else:
        out_shape = (n_levels,)
    try:
        if T.shape != q.shape:
            raise Exception(
                """get_gz: T and q arguments must have the same
                            shape"""
            )
    except:
        raise Exception(
            """get_gz: Cannot get shape attribute of T and/or q
                        arguments"""
        )
    else:
        for dim in T.shape:
            if dim not in out_shape:
                raise Exception(
                    """get_gz: Shape of T not compatible with ps and
                                and gzs arguments"""
                )
        for dim in q.shape:
            if dim not in out_shape:
                raise Exception(
                    """get_gz: Shape of T not compatible with ps and
                                and gzs arguments"""
                )

    # Find axis index of height dimension
    h_axis = [i for i, ax in enumerate(out_shape) if ax not in ps_shape][0]
    ps_reshape = list(out_shape)
    ps_reshape[h_axis] = 1
    ps_reshape = tuple(ps_reshape)

    # Get half level pressure, full level pressure and delta p
    ph = get_ph(ps, n_levels)
    pl = get_pl(ps, n_levels)
    dp = get_dp(ps, n_levels)
    # dlogP -- change in logP
    dlogP = np.zeros(out_shape)
    dlogP[1:] = np.log(ph[2:] / ph[1:-1])
    dlogP[0] = np.log(ph[1] / pl[0])
    # alpha factor for calculating geopotential
    alpha = 1.0 - (ph[1:] / dp) * dlogP
    alpha[0] = np.log(2)
    # Get moist temperature at each level
    Tm = T * (1.0 + 0.609133 * q)
    TRd = Tm * 287.06
    # Get geopotential half levels
    gzh = np.cumsum((TRd * dlogP)[::-1], axis=h_axis)[::-1] + gzs
    gzf = gzh + TRd * alpha
    return gzf


def filter_sel(sel: dict) -> dict:
    if "level" in sel:
        if isinstance(sel["level"], slice):
            if sel["level"].start is None:
                sel["level"] = slice(1, sel["level"].stop, sel["level"].step)
            if sel["level"].stop is None:
                sel["level"] = slice(sel["level"].start, 137, sel["level"].step)
    if "longitude" in sel:
        if isinstance(sel["longitude"], slice):
            if sel["longitude"].start is None:
                sel["longitude"] = slice(
                    0, sel["longitude"].stop, sel["longitude"].step
                )
            if sel["longitude"].stop is None:
                sel["longitude"] = slice(
                    sel["longitude"].start, 360, sel["longitude"].step
                )
    if "latitude" in sel:
        if (
            isinstance(sel["latitude"], slice)
            and sel["latitude"].stop > sel["latitude"].start
        ):
            sel["latitude"] = slice(
                sel["latitude"].stop, sel["latitude"].start, sel["latitude"].step
            )
        if isinstance(sel["latitude"], slice):
            if sel["latitude"].start is None:
                sel["latitude"] = slice(90, sel["latitude"].stop, sel["latitude"].step)
            if sel["latitude"].stop is None:
                sel["latitude"] = slice(sel["latitude"].stop, -90, sel["latitude"].step)
    return sel


def sel_era5(ds, sel):
    sel = filter_sel(sel)
    sel_coords = {k: v for k, v in sel.items() if k in ds.coords}
    # To slice over the prime meridian we have to be a little more clever
    if "longitude" in sel and (
        sel["longitude"].start < 0 or sel["longitude"].start > sel["longitude"].stop
    ):
        ds = ds.sel({k: v for k, v in sel_coords.items() if k != "longitude"})
        start_lon = sel["longitude"].start % 360
        stop_lon = sel["longitude"].stop % 360
        start_idx = ((ds.longitude - start_lon) % 360).argmin().item()
        stop_idx = ((ds.longitude - stop_lon) % 360).argmin().item()
        if ds.longitude[stop_idx] == stop_lon:
            stop_idx += 1
        stop_idx = (stop_idx - start_idx) % 1440
        ds = ds.roll(longitude=-start_idx, roll_coords=True).isel(
            longitude=slice(0, stop_idx)
        )
    else:
        ds = ds.sel(sel_coords)
    return ds
