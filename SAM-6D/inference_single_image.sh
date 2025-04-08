export CAD_PATH=/home/ohseun/workspace/SAM-6D/SAM-6D/Data/BOP/tless/models_cad/obj_000029.ply    # path to a given cad model(mm)
export RGB_PATH=/home/ohseun/workspace/SAM-6D/SAM-6D/Data/BOP/tless/test_primesense/000001/rgb/000101.png          # path to a given RGB image
export DEPTH_PATH=/home/ohseun/workspace/SAM-6D/SAM-6D/Data/BOP/tless/test_primesense/000001/depth/000101.png       # path to a given depth map(mm)
export CAMERA_PATH=/home/ohseun/workspace/SAM-6D/SAM-6D/Data/BOP/tless/test_primesense/000001/scene_camera.json    # path to given camera intrinsics
export OUTPUT_DIR=/home/ohseun/workspace/SAM-6D/SAM-6D/Data/BOP/tless/outputs 

cd Instance_Segmentation_Model

# # Run instance segmentation model
export SEGMENTOR_MODEL=sam

# python run_inference_custom.py --segmentor_model $SEGMENTOR_MODEL --output_dir $OUTPUT_DIR --cad_path $CAD_PATH --rgb_path $RGB_PATH --depth_path $DEPTH_PATH --cam_path $CAMERA_PATH

cd ../Pose_Estimation_Model

# Run pose estimation model
export SEG_PATH=$OUTPUT_DIR/sam6d_results/101_detection_ism.json

cd ../Pose_Estimation_Model
python run_inference_custom.py --output_dir $OUTPUT_DIR --cad_path $CAD_PATH --rgb_path $RGB_PATH --depth_path $DEPTH_PATH --cam_path $CAMERA_PATH --seg_path $SEG_PATH

cd ..