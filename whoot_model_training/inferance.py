"""Trains a Mutliclass Model with Pytorch and Huggingface.

This script can be used to run experiments with different
models and datasets to create any model for bioacoustic classification

It is intended this script to be heavily modified with each experiment
(say one wants to use a different dataset, one should copy this and change the
extractor!)

Usage:
    $ python train.py /path/to/config.yml

config.yml should contain frequently changed hyperparameters
"""
import argparse
import pickle
import datasets

from train import parse_config, init_env


from whoot_model_training.preprocessors.base_preprocessor import WaveformInputPreprocessor
from whoot_model_training.trainer import WhootTrainer, WhootTrainingArguments
from whoot_model_training.data_extractor import raw_audio_extractor
from whoot_model_training.models import TimmModel, TimmInputs
from whoot_model_training.models import HFInput, HFModel, HFModelConfig
from whoot_model_training import CometMLLoggerSupplement
from whoot_model_training.preprocessors import (
    MelModelInputPreprocessor
)




def test(
    config,
    model_name="",
    audio_dir="/mnt/restorage/Audiomoth/Raw sound files/2024/RGCB/"
):
    """Highest level logic for inferance.

    Does the following:
    - Formats the dataset into an AudioDataset
    - Prepares preprocessing for each audio clip
    - Builds the model
    - Configures and runs the trainer
    - Runs evaluation

    Args:
        config (dict): the config used for training. Defined in yaml file
        model_name (str): path to model checkpoint to use
        audio_dir (str): path to unlabeled data
    """
    # Extract a new dataset
    ds = raw_audio_extractor(
        audio_parent_folder=audio_dir,
        output_folder="data/manual_buowset",
        chunk_duration=3,
        class_list=["Abert's Towhee", 'Acorn Woodpecker', "Allen's Hummingbird", 'American Avocet', 'American Barn Owl', 'American Bittern', 'American Bullfrog', 'American Bushtit', 'American Cliff Swallow', 'American Coot', 'American Crow', 'American Dusky Flycatcher', 'American Goldfinch', 'American Grey Flycatcher', 'American Herring Gull', 'American Kestrel', 'American Redstart', 'American Robin', 'American Wigeon', 'American Yellow Warbler', "Anna's Hummingbird", 'Ash-throated Flycatcher', "Audubon's Warbler", 'Band-tailed Pigeon', 'Barn Swallow', 'Bay-breasted Warbler', "Bell's Sparrow", "Bell's Vireo", 'Belted Kingfisher', "Bewick's Wren", 'Black Phoebe', 'Black Skimmer', 'Black Turnstone', 'Black-chinned Hummingbird', 'Black-chinned Sparrow', 'Black-crowned Night Heron', 'Black-headed Grosbeak', 'Black-hooded Oriole', 'Black-necked Grebe', 'Black-necked Stilt', 'Black-tailed Gnatcatcher', 'Black-throated Grey Warbler', 'Black-throated Magpie-Jay', 'Black-throated Sparrow', 'Blue Grosbeak', 'Blue-crowned Parakeet', 'Blue-grey Gnatcatcher', "Bonaparte's Gull", "Brandt's Cormorant", 'Brant Goose', "Brewer's Blackbird", "Brewer's Sparrow", 'Brown Creeper', 'Brown-headed Cowbird', 'Buff-bellied Pipit', "Bullock's Oriole", 'Burrowing Owl', 'Burrowing Parrot', 'Cactus Wren', 'California Gnatcatcher', 'California Ground Squirrel', 'California Gull', 'California Quail', 'California Scrub Jay', 'California Thrasher', 'California Towhee', 'Canada Goose', 'Canada Warbler', 'Canyon Bat', 'Canyon Wren', 'Caspian Tern', "Cassin's Finch", "Cassin's Kingbird", "Cassin's Vireo", 'Cedar Waxwing', 'Chestnut-collared Longspur', 'Chipping Sparrow', 'Cinnamon Teal', 'Cinnamon-rumped Seedeater', "Clark's Grebe", "Clark's Nutcracker", 'Clay-colored Sparrow', 'Cockatiel', 'Common Gallinule', 'Common Ground Dove', 'Common Poorwill', 'Common Starling', 'Common Yellowthroat', "Cooper's Hawk", "Costa's Hummingbird", 'Coyote', "Craveri's Murrelet", 'Crissal Thrasher', 'Dark-eyed Junco', 'Double-crested Cormorant', 'Downy Woodpecker', 'Dunlin', 'Dusky-capped Flycatcher', 'Eastern Subalpine Warbler', 'Elegant Tern', 'Eurasian Collared Dove', 'Evening Grosbeak', "Forster's Tern", 'Gadwall', "Gambel's Quail", 'Gila Woodpecker', 'Glaucous-blue Grosbeak', 'Glaucous-winged Gull', 'Golden-crowned Kinglet', 'Golden-crowned Sparrow', "Grace's Warbler", 'Grasshopper Sparrow', 'Great Blue Heron', 'Great Egret', 'Great Horned Owl', 'Great-tailed Grackle', 'Greater Pewee', 'Greater Roadrunner', 'Greater Yellowlegs', 'Green-tailed Towhee', 'Green-winged Teal', 'Grey Catbird', 'Grey Plover', 'Grey Vireo', 'Grey-hooded Warbler', 'Gull-billed Tern', 'Hairy Woodpecker', "Hammond's Flycatcher", "Heermann's Gull", 'Hermit Thrush', 'Hermit Warbler', 'Hooded Oriole', 'Hooded Warbler', 'Horned Lark', 'House Finch', 'House Sparrow', 'House Wren', 'Hudsonian Whimbrel', "Hutton's Vireo", 'Identity unknown', 'Inca Dove', 'Indian House Cricket', 'Killdeer', 'Lapland Longspur', 'Lark Sparrow', 'Laughing Gull', "Lawrence's Goldfinch", 'Lazuli Bunting', "LeConte's Thrasher", 'Least Bittern', 'Least Sandpiper', 'Least Tern', 'Lesser Goldfinch', 'Lesser Nighthawk', 'Lilac-crowned Amazon', "Lincoln's Sparrow", 'Little Blue Heron', 'Loggerhead Shrike', 'Long-billed Curlew', 'Long-billed Dowitcher', 'Long-chirp field cricket', 'Louisiana Waterthrush', "Lucy's Warbler", "MacGillivray's Warbler", 'Mallard', 'Marbled Godwit', 'Marsh Wren', "Merriam's Chipmunk", 'Mexican Whip-poor-will', 'Mountain Bluebird', 'Mountain Chickadee', 'Mountain Quail', 'Mourning Dove', 'Nashville Warbler', "Nelson's Sparrow", 'Northern Flicker', 'Northern Harrier', 'Northern Mockingbird', 'Northern Parula', 'Northern Pintail', 'Northern Raven', 'Northern Rough-winged Swallow', 'Northern Saw-whet Owl', 'Northern Shoveler', 'Northern Waterthrush', "Nuttall's Woodpecker", 'Oak Titmouse', 'Olive-sided Flycatcher', 'Orange-crowned Warbler', 'Pacific Golden Plover', 'Pacific Treefrog', 'Pacific Wren', 'Pacific-slope Flycatcher', 'Palm Warbler', 'Pelagic Cormorant', 'Peregrine Falcon', 'Phainopepla', 'Pied-billed Grebe', 'Pin-tailed Whydah', 'Pine Siskin', 'Pine Warbler', 'Pinyon Jay', 'Plumbeous Vireo', 'Prairie Warbler', 'Purple Finch', 'Purple Martin', 'Pygmy Nuthatch', 'Red Crossbill', 'Red-breasted Nuthatch', 'Red-crowned Amazon', 'Red-crowned Crane', 'Red-eyed Vireo', 'Red-faced Warbler', 'Red-masked Parakeet', 'Red-naped Sapsucker', 'Red-necked Grebe', 'Red-necked Phalarope', 'Red-shouldered Hawk', 'Red-tailed Hawk', 'Red-throated Pipit', 'Red-winged Blackbird', 'Redhead', "Ridgway's Rail", 'Ring-billed Gull', 'Rock Dove', 'Rock Wren', 'Rose-breasted Grosbeak', "Ross's Goose", 'Round-tailed Ground Squirrel', 'Royal Tern', 'Ruby-crowned Kinglet', 'Ruddy Duck', 'Rufous Hummingbird', 'Rufous-crowned Sparrow', 'Rusty Blackbird', 'Sage Thrasher', 'Sanderling', 'Sandhill Crane', 'Savannah Sparrow', "Say's Phoebe", 'Scaly-breasted Munia', "Scott's Oriole", 'Sharp-shinned Hawk', 'Short-billed Dowitcher', 'Slate-colored Fox Sparrow', 'Snow Goose', 'Snow Mountain Quail', 'Snowy Egret', 'Snowy Plover', 'Solitary Sandpiper', 'Song Sparrow', 'Sooty Fox Sparrow', 'Sora', 'Soundscape', 'Spotted Sandpiper', 'Spotted Towhee', "Steller's Jay", 'Stilt Sandpiper', 'Summer Tanager', 'Surf Scoter', 'Surfbird', "Swainson's Thrush", "Swinhoe's White-eye", 'Tennessee Warbler', 'Thick-billed Fox Sparrow', 'Thick-billed Kingbird', 'Thick-billed Longspur', "Townsend's Solitaire", "Townsend's Warbler", 'Tree Swallow', 'Tricolored Blackbird', 'Tropical Kingbird', 'Two-barred Crossbill', 'Verdin', 'Vermilion Flycatcher', 'Violet-green Swallow', 'Virginia Rail', 'Vocal field cricket', 'Wandering Tattler', 'Warbling Vireo', 'Western Bluebird', 'Western Cattle Egret', 'Western Grebe', 'Western Gull', 'Western Kingbird', 'Western Meadowlark', 'Western Osprey', 'Western Sandpiper', 'Western Screech Owl', 'Western Subalpine Warbler', 'Western Tanager', 'Western Wood Pewee', 'White-breasted Nuthatch', 'White-crowned Sparrow', 'White-eyed Vireo', 'White-faced Ibis', 'White-tailed Kite', 'White-throated Sparrow', 'White-throated Swift', 'White-winged Dove', 'Wild Turkey', 'Willet', 'Willow Flycatcher', "Wilson's Snipe", "Wilson's Warbler", 'Wood Duck', 'Wrentit', 'Yellow-breasted Chat', 'Yellow-crowned Night Heron', 'Yellow-footed Gull']
    )

    # ds = buowset_extractor(
    #     metadata_csv=config["metadata_csv"],
    #     parent_path=config["data_path"],
    #     output_path=config["hf_cache_path"],
    # )

    # Create the model
    
    model = HFModel.from_pretrained(model_name).cuda()

    # %%
    # %%

    input_wrapper = HFInput()

    train_preprocessor = WaveformInputPreprocessor(
        input_wrapper, duration=3
    )


    ds["train"].set_transform(train_preprocessor)
    # ds["valid"].set_transform(preprocessor)
    # ds["test"].set_transform(preprocessor)

    model_name = "efficientnet_b1"
    run_name = f"fewshot_test_birdmae_11_12_2025_14_checkpoint-137500"

    # trainer = WhootTrainer._load_from_checkpoint(model_name)
    # Run training
    training_args = WhootTrainingArguments(
        run_name=run_name,
        subproject_name=config["SUBPROJECT_NAME"]+"_INFERANCE",
        dataset_name=config["DATASET_NAME"],
    )

    # COMMON OPTIONAL ARGS
    training_args.num_train_epochs = 5
    training_args.eval_steps = 100
    training_args.per_device_train_batch_size = 16
    training_args.per_device_eval_batch_size = 16
    training_args.dataloader_num_workers = 1
    training_args.run_name = run_name

    trainer = WhootTrainer(
        model=model,
        dataset=ds,
        training_args=training_args,
        logger=CometMLLoggerSupplement(
            augmentations=None,
            name=training_args.run_name
        ),
    )

    # print(ds["train"].shape, ds["test"].shape, ds["valid"].shape)
    # input()

    out = trainer.predict(ds["train"], save_path=f"predictions/{run_name}")
    # Pipeline requires a labels col
    # For inferance the "labels" are just an array of zeros
    # Therefore during inferance, "labels" are meaningless
    # Delete them to make it clearer to downstream users
    del out['labels']

    with open(run_name + ".pkl", mode="wb") as f:
        pickle.dump(out, f)
    # Below was tested with the pickle made from above
    ds = datasets.Dataset.from_dict(out)
    ds.save_to_disk(f"predictions/{run_name}") # saves as a directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input config path")
    parser.add_argument("config", type=str, help="Path to config.yml")
    parser.add_argument(
        "--model_name",
        required=False,
        help="path to weights or hugging face repo id",
        default="/home/sean/whoot/model_checkpoints/fewshot_test_birdmae_11_12_2025_14:04:09/checkpoint-137500")
    args = parser.parse_args()
    _config = parse_config(args.config)

    init_env(_config)
    test(_config, model_name=args.model_name)
