mkdir -p deepfashion-mm
cd deepfashion-mm

# Textual description
if [ ! -f "captions.json" ]
then
    gdown --fuzzy https://drive.google.com/file/d/1d1TRm8UMcQhZCb6HpPo8l3OPEin4Ztk2/view?usp=sharing
fi


# Textual labels
if [ ! -d "./labels" ]
then
    gdown --fuzzy https://drive.google.com/file/d/11WoM5ZFwWpVjrIvZajW0g8EmQCNKMAWH/view?usp=sharing
    unzip labels.zip
    rm labels.zip
fi


# Images jpg
if [ ! -d "./images" ]
then
    gdown --fuzzy https://drive.google.com/file/d/1U2PljA7NE57jcSSzPs21ZurdIPXdYZtN/view?usp=sharing
    unzip images.zip
    rm images.zip
fi