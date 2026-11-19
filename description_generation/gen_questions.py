import os
import argparse
import yaml
from pathlib import Path
from dotenv import load_dotenv

from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

client = OpenAI(
    timeout=60.0,
    max_retries=2
)


DATASET2DESC = {
    "imagenet": "an image database containing millions of images across thousands of categories",
    "imagenet_r": "contains different reditions, in forms of: art, cartoons, deviantart, graffiti, embroidery, graphics, origami, paintings, patterns, plastic objects, plush objects, sculptures, sketches, tattoos, toys, and video game renditions",
    "imagenet_sketch": "consists of black and white sketches of ImageNet categories",
    "caltech101": "contains images from 101 object categories",
    "oxford_pets": "a pet dataset whose images have a large variations in scale, pose and lighting.",
    "stanford_cars": "contains images of cars whose classes are typically at the level of Make, Model, Year, ex.",
    "oxford_flowers": "the flowers chosen to be flower commonly occuring in the United Kingdom with large scale, pose and light variations",
    "food101": "consists of 101 food categories with some amount of noise",
    "fgvc_aircraft": "contains images of different aircraft model variants, most of which are airplanes",
    "sun397": "a Scene UNderstanding dataset with 397 categories",
    "dtd": "has collection of textural images in the wild",
    "eurosat": "contains satellite view, based on Sentinel-2 satellite images for land use and land cover classification",
    "ucf101": "an action recognition data set of realistic action videos",
    
    "caltech256": "an object recognition dataset containing real-world images",
    "cub": "a challenging dataset of 200 bird species",
    "birdsnap": "a large bird dataset with 500 bird species",
    
    "cifar10":  "a dataset consists of photo-realistic color images in a resolution of only 32x32",
    "cifar100": "a dataset consists of photo-realistic color images in a resolution of only 32x32",
    "country211": "a dataset containing photos taken in different countries to evaluate model's geolocation capability",
    
    "hmdb51": "a large collection of realistic videos from various sources",
    "ucf101v": "an action recognition data set of realistic action videos",
    "k600": "a large-scale action recognition dataset which consists of around 480K videos from 600 action categories",
    "household_objects": (
        "contains 100 photographs of everyday household objects "
        "with variations in appearance, shape, materials, and lighting"
    ),
}


def process_answer(answer):
    lines = [line.strip('- ') for line in answer.split('\n')]
    return lines


def get_dataset_questions(args, dataset_name=''):
    
    assert args.llm in ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-2024-08-06", "qwen-plus"]
    
    SYS_PROMPT ="You are a helpful Q&A assistant. You need to generate questions for LLM to describe objects. "\
                "These questions should be open-ended and aimed at eliciting visually descriptive features. "\
                "Consider questions related to visually identifying the object or describing images of it. "\
                "Any features that cannot be visually classified like smell, temperature should not appear. "\
                "Questions like 'A painting of a {}' or 'What does a {} look like?' would be suitable. "\
                "Your questions should depend on the given category like 'pets' or 'foods'. "\
                "No actual types of objects should appear in questions. Use an empty {} instead. {} should appear once and only once. "\
                "Use the category name like 'buildings' if the class names to be classified are unclear. "\
                "Do pay attention to the dataset desciption. Use the form of the images in your questions."
    SYS_PROMPT_NONVISUAL = (
        "You are a helpful Q&A assistant. "
        "You need to generate questions for an LLM to describe objects. "
        "These questions should be open-ended and focus on non-visual "
        "semantic information about objects. "
        "Consider questions about purpose, function, typical uses, "
        "common settings, activities, and associations. "
        "Avoid questions about color, shape, appearance, texture, "
        "material, lighting, or other directly visible features. "
        "Your questions should depend on the given dataset category. "
        "No actual types of objects should appear in questions. "
        "Use an empty {} instead of an object name. "
        "The placeholder {} should appear once and only once "
        "in each question."
    )

    USR_QUERY_TEMPLATE_NODESC = 'Generate questions to classify images from a dataset.'
    USR_QUERY_TEMPLATE_DESC = 'Generate questions to classify images from a dataset, which {}.'
    AGENT_ANSWER_EXAMPLE =  "- What distinct features can you see on {}?\n"                     \
                            "- How would you describe the texture in {}?\n"                     \
                            "- What patterns or designs are visible on {}?\n"                   \
                            "- What colors are prominent in {}?\n"                              \
                            "- Describe the shape and structure of {}.\n"                       \
                            "- What elements in {} give you a sense of its size or scale?\n"    \
                            "- What unique characteristics stand out on {}?\n"                  \
                            "- Are there any distinguishing characteristics in the {} that help in its categorization?" 
    AGENT_ANSWER_EXAMPLE_NONVISUAL = (
        "- What is the primary purpose of {}?\n"
        "- What activities commonly involve {}?\n"
        "- In what situations is {} typically used?\n"
        "- What practical needs does {} fulfill?\n"
        "- Where is {} commonly used?\n"
        "- What cultural or everyday associations are connected to {}?\n"
        "- What role does {} play in daily life?\n"
        "- What other objects are commonly used alongside {}?"
    )


    system_prompt = (
        SYS_PROMPT_NONVISUAL
        if args.condition == "D"
        else SYS_PROMPT
    )

    messages = []
    messages.append({
        "role": "system",
        "content": system_prompt
    })
    """messages = []
    messages.append({"role": "system", "content": SYS_PROMPT})""" # Replaced to enable condition D
    if args.no_dataset_desc:
        messages.append({"role": "user", "content": USR_QUERY_TEMPLATE_NODESC})
    else:
        messages.append({"role": "user", "content": USR_QUERY_TEMPLATE_DESC.format(DATASET2DESC["imagenet"])})
    """messages.append({"role": "assistant", "content": AGENT_ANSWER_EXAMPLE})""" # Replaced to enable condition D
    messages.append({
        "role": "assistant",
        "content": (
            AGENT_ANSWER_EXAMPLE_NONVISUAL
            if args.condition == "D"
            else AGENT_ANSWER_EXAMPLE
        )
    })
    if args.no_dataset_desc:
        messages.append({"role": "user", "content": USR_QUERY_TEMPLATE_NODESC})
    else:
        messages.append({"role": "user", "content": USR_QUERY_TEMPLATE_DESC.format(DATASET2DESC[dataset_name])})
    """if args.no_dataset_desc:
        messages.append({"role": "user", "content": USR_QUERY_TEMPLATE_NODESC})
    else:
        messages.append({"role": "user", "content": USR_QUERY_TEMPLATE_DESC.format(DATASET2DESC["eurosat"])})""" # Used for condition C
    
    response = client.chat.completions.create(
        model=args.llm,
        messages=messages,
        temperature=0.5
    )
    answer = response.choices[0].message.content
    ret = process_answer(answer)
    return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm', type=str, default="gpt-4o-2024-08-06", help='The LLM-API to use for generation.')
    parser.add_argument('--prompt_folder', type=str, default='Prompts', help='The root folder to save all generated prompts.')
    parser.add_argument('--question_file', type=str, default='0_Dataset_Questions.yaml', help='The file name of generated Step-1 dataset questions.')
    parser.add_argument('--no_dataset_desc', action='store_true', help='not using dataset description for generation')
    parser.add_argument(
        "--condition",
        choices=["A", "B", "C", "D"],
        default="A"
    )
    args = parser.parse_args()

    data = {}

    ret = get_dataset_questions(args, "household_objects")
    data["household_objects"] = ret
    print("household_objects:", ret)

    save_dir = os.path.join(args.prompt_folder, args.llm)
    save_path = os.path.join(save_dir, args.question_file)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_path, 'w') as f:
        yaml.dump(data, f, width=1024)
