// Calls the official CMU Skeleton::build_pose and get_world_bones unchanged.
// Only the small loader/export harness is ours. No viewer, OpenGL or SDK.
#include <Library/Skeleton.hpp>
#include <Library/ReadSkeleton.hpp>
#include <Character/pose_utils.hpp>
#include <fstream>
#include <iostream>
#include <map>
#include <iomanip>

int main(int argc,char **argv) {
    if(argc!=6)return 2;
    Library::Skeleton sk;
    if(!ReadSkeleton(argv[1],sk))return 3;
    std::vector<double> frames;
    if(!ReadAnimation(argv[2],sk,frames))return 4;
    unsigned int B=sk.bones.size(),N=frames.size()/sk.frame_size;double unit=sk.length;
    if(B!=30||N!=940||unit!=0.45||sk.frame_size!=62||sk.rot_is_glob||sk.z_is_up)return 5;
    std::ofstream out(argv[3],std::ios::binary),names(argv[4]),channels(argv[5],std::ios::binary);
    if(!out||!names||!channels)return 6;
    names<<std::setprecision(17)<<unit<<" "<<sk.frame_size<<" "<<sk.order<<"\n";
    for(unsigned int b=0;b<B;++b) {
        const Library::Bone &bone=sk.bones[b];
        names<<bone.name<<" "<<(bone.parent<0?"root":sk.bones[bone.parent].name)<<" "<<bone.frame_offset<<" "
            <<(bone.dof.empty()?"-":bone.dof)<<" "<<bone.length<<" "<<bone.direction.x<<" "<<bone.direction.y<<" "<<bone.direction.z<<" "
            <<bone.axis_offset.x<<" "<<bone.axis_offset.y<<" "<<bone.axis_offset.z<<" "<<bone.offset_order<<"\n";
    }
    const double viewer_to_meters=0.0254/(unit*unit);
    for(unsigned int f=0;f<N;++f) {
        double *frame=&frames[f*sk.frame_size];
        channels.write(reinterpret_cast<char*>(frame),sizeof(double)*sk.frame_size);
        Character::Pose pose;sk.build_pose(frame,pose);Character::WorldBones wb;
        Character::get_world_bones(pose,wb);
        double root[3]={pose.root_position.x*viewer_to_meters,pose.root_position.y*viewer_to_meters,pose.root_position.z*viewer_to_meters};
        out.write(reinterpret_cast<char*>(root),sizeof(root));
        for(unsigned int b=0;b<B;++b) {
            double data[10]={wb.bases[b].x*viewer_to_meters,wb.bases[b].y*viewer_to_meters,wb.bases[b].z*viewer_to_meters,
                wb.tips[b].x*viewer_to_meters,wb.tips[b].y*viewer_to_meters,wb.tips[b].z*viewer_to_meters,
                wb.orientations[b].x,wb.orientations[b].y,wb.orientations[b].z,wb.orientations[b].w};
            out.write(reinterpret_cast<char*>(data),sizeof(data));
        }
    }
    out.close();names.close();channels.close();if(!out||!names||!channels)return 8;
    std::cout<<"official_CMU_native_frames="<<N<<" bones="<<B<<"\n";return 0;
}
