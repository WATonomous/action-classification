# https://github.com/gurkirt/3D-RetinaNet/blob/7b9d28c703b9af8e67d9d18dc7dd91b1c4d2d91a/kinetics-pt/get_kinetics_weights.sh
#Downloading Kinetics weights
wget --load-cookies cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=17xiC_Wrdv1noD9NZmgXGIZQWSnW0wnxP' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=17xiC_Wrdv1noD9NZmgXGIZQWSnW0wnxP" -O resnet50C2D.pth
rm cookies.txt
wget --load-cookies cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1XBMs4TLt2H378M_a0k23l8let-Ae2AlB' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1XBMs4TLt2H378M_a0k23l8let-Ae2AlB" -O resnet50I3D-NL.pth
rm cookies.txt
wget --load-cookies cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ZpbvJzvnDxJmKCFTs9wKmmA2qvm2aFBX' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ZpbvJzvnDxJmKCFTs9wKmmA2qvm2aFBX" -O resnet50I3D.pth
rm cookies.txt
