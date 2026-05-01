import subprocess

all_paths = {
    'JAL004': [
        '004_baseline_2023_08_17T13_41_44', 
        'JAL004_flip_rotated_2023_08_28T09_36_04',
        '004_flip_puff2_2023_09_11T09_32_25',
        '004_flip_2023_09_03T12_04_16'
    ],

    'JAL005': [
        '005_baseline_2023_09_05T07_48_58',
        '005_flip1_2023_09_08T07_36_54',
        '005_flippuff3_2023_09_21T11_11_13',
    ],

    'JAL006': [
        'JAL006_barrier_flip2_2024_03_18T11_53_29',
        'JAL006_shelter_barrier_flip_3_2024_03_21T11_20_34',
        'JAL006_shelter_barrier_flip_5_2024_03_25T11_05_33',
        'JAL006_shelter_barrier_flip_6_2024_03_28T10_54_20',
    ],

    'JAL007': [
        'JAL007_barrierflip2_2024_03_12T11_18_26',
        'JAL007_shelter_barrier_flip_5_2024_03_22T11_15_43',
        'JAL007_shelter_barrier_flip_8_2024_04_09T10_07_45',
        'JAL007_shelter_barrier_flip_9_2024_04_16T11_13_05',
        'JAL007_shelter_barrier_flip_100_2024_04_23T09_59_40',
        'JAL007_tinnybarrier1_2024_04_30T10_57_04',
    ],

    'JAL008': [
        'JAL008_shelter_barrier_flip_1_2024_04_25T11_27_42',
        'JAL008_shelter_barrier_flip_2_2024_04_29T12_14_54',
        'JAL008_shelter_tiny_barrier_flip_1_2024_05_03T10_02_35',
        'JAL008_shelter_barrier_flip_3_2024_05_07T10_16_26',
        'JAL008_shelter_barrier_flip_4_2024_05_10T11_47_47',
        'JAL008_shelter_barrier_flip_5_2024_05_14T10_18_03',
        'JAL008_shelter_tiny_flip_2_2024_05_21T11_10_19'
    ]      
}



def enumerate_paths():
    for animal_idx, (animal, animal_exp_paths) in enumerate(all_paths.items()):
        for exp_idx, exp_path in enumerate(animal_exp_paths):
            yield animal_idx, exp_idx, animal, exp_path


if __name__ == '__main__':

    for animal_idx, exp_idx, animal, exp_path in enumerate_paths():

        subprocess.run(f'bash/run_cpu.sh gen-data data-scripts/generate_data.py {animal_idx} {exp_idx} {animal} {exp_path}', shell=True, capture_output=False)
