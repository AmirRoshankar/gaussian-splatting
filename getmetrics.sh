python render.py --model_path output/masked_colour_and_geom_1_0/ --skip_train --overwrite
python render.py --model_path output/masked_colour_and_geom_100_1/ --skip_train --overwrite
python render.py --model_path output/masked_colour_and_geom_10_1/ --skip_train --overwrite
python render.py --model_path output/masked_colour_and_geom_1_1/ --skip_train --overwrite
python render.py --model_path output/masked_colour_and_geom_1_10/ --skip_train --overwrite
python render.py --model_path output/masked_colour_and_geom_1_100/ --skip_train --overwrite
python metrics.py --model_path output/masked_colour_and_geom_1_0/
python metrics.py --model_path output/masked_colour_and_geom_100_1/
python metrics.py --model_path output/masked_colour_and_geom_10_1/
python metrics.py --model_path output/masked_colour_and_geom_1_1/
python metrics.py --model_path output/masked_colour_and_geom_1_10/
python metrics.py --model_path output/masked_colour_and_geom_1_100/