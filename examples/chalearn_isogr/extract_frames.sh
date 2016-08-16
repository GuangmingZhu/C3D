#!/bin/bash
ROOTDIR=/ssd/dataset

[[ -d $ROOTDIR/IsoGD_Image ]] || mkdir $ROOTDIR/IsoGD_Image
# 1. Convert train videos to images
VIDEOPATH=$ROOTDIR/IsoGD_phase_1/train
IMAGEPATH=$ROOTDIR/IsoGD_Image/train_image
[[ -d $IMAGEPATH ]] || mkdir $IMAGEPATH
[[ -d $IMAGEPATH/RGB ]] || mkdir $IMAGEPATH/RGB
[[ -d $IMAGEPATH/Depth ]] || mkdir $IMAGEPATH/Depth
for dir in `ls $VIDEOPATH`
do
	VIDEOSUBPATH=$VIDEOPATH/$dir
	for file in `ls $VIDEOSUBPATH/K_*.avi`
	do
		FILENAME=`basename $file`
		NAME=${FILENAME%.*}
		BNAME=$IMAGEPATH/RGB/$NAME
		mkdir -m 755 $BNAME
		ffmpeg -i $file $BNAME/%06d.jpg
	done
	for file in `ls $VIDEOSUBPATH/M_*.avi`
	do
		FILENAME=`basename $file`
		NAME=${FILENAME%.*}
		BNAME=$IMAGEPATH/Depth/$NAME
		mkdir -m 755 $BNAME
		ffmpeg -i $file $BNAME/%06d.jpg
	done
done

# 2. Convert valid videos to images
VIDEOPATH=$ROOTDIR/IsoGD_phase_1/valid
IMAGEPATH=$ROOTDIR/IsoGD_Image/valid_image
[[ -d $IMAGEPATH ]] || mkdir $IMAGEPATH
[[ -d $IMAGEPATH/RGB ]] || mkdir $IMAGEPATH/RGB
[[ -d $IMAGEPATH/Depth ]] || mkdir $IMAGEPATH/Depth
for dir in `ls $VIDEOPATH`
do
	VIDEOSUBPATH=$VIDEOPATH/$dir
	for file in `ls $VIDEOSUBPATH/K_*.avi`
	do
		FILENAME=`basename $file`
		NAME=${FILENAME%.*}
		BNAME=$IMAGEPATH/RGB/$NAME
		mkdir -m 755 $BNAME
		ffmpeg -i $file $BNAME/%06d.jpg
	done
	for file in `ls $VIDEOSUBPATH/M_*.avi`
	do
		FILENAME=`basename $file`
		NAME=${FILENAME%.*}
		BNAME=$IMAGEPATH/Depth/$NAME
		mkdir -m 755 $BNAME
		ffmpeg -i $file $BNAME/%06d.jpg
	done
done

# 3. Convert test videos to images
VIDEOPATH=$ROOTDIR/IsoGD_phase_2/test
IMAGEPATH=$ROOTDIR/IsoGD_Image/test_image
[[ -d $IMAGEPATH ]] || mkdir $IMAGEPATH
[[ -d $IMAGEPATH/RGB ]] || mkdir $IMAGEPATH/RGB
[[ -d $IMAGEPATH/Depth ]] || mkdir $IMAGEPATH/Depth
for dir in `ls $VIDEOPATH`
do
	VIDEOSUBPATH=$VIDEOPATH/$dir
	for file in `ls $VIDEOSUBPATH/K_*.avi`
	do
		FILENAME=`basename $file`
		NAME=${FILENAME%.*}
		BNAME=$IMAGEPATH/RGB/$NAME
		mkdir -m 755 $BNAME
		ffmpeg -i $file $BNAME/%06d.jpg
	done
	for file in `ls $VIDEOSUBPATH/M_*.avi`
	do
		FILENAME=`basename $file`
		NAME=${FILENAME%.*}
		BNAME=$IMAGEPATH/Depth/$NAME
		mkdir -m 755 $BNAME
		ffmpeg -i $file $BNAME/%06d.jpg
	done
done
