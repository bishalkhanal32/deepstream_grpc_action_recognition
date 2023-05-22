// Copyright 2020 - NVIDIA Corporation
// SPDX-License-Identifier: MIT

#include "post_process.cpp"
#include "deepstream_action.h"


#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>

#include "gstnvdsmeta.h"
#include "nvdsgstutils.h"
#include "nvbufsurface.h"
// for fps calculation
#include "deepstream_perf.h"


// grpc part
#include "detector.grpc.pb.h"
#include <iostream>
#include <memory>
#include <grpcpp/grpcpp.h>


#include <vector>
#include <array>
#include <queue>
#include <cmath>
#include <string>
#include <algorithm>

#define TRACKER_CONFIG_FILE "/opt/nvidia/deepstream/deepstream/sources/apps/Deepstream-app/deepstream_yolo_vitpose_grpc/dstest2_tracker_config.txt"

#define NVDS_USER_META (nvds_get_user_meta_type("NVIDIA.NVINFER.USER_META"))



#define EPS 1e-6
#define GST_CAPS_FEATURES_NVMM "memory:NVMM"

#define MAX_DISPLAY_LEN 64

// for fps calculation from https://github.com/NVIDIA-AI-IOT/deepstream_pose_estimation/pull/3/files#diff-3ce9c925281203605c3d7ad6876b93eab676fdff1158d3bd82ffab0fe32591e5
#define MAX_STREAMS 64


/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH gActionConfig.muxer_width
#define MUXER_OUTPUT_HEIGHT gActionConfig.muxer_height

/* By default, OSD process-mode is set to CPU_MODE. To change mode, set as:
 * 1: GPU mode (for Tesla only)
 * 2: HW mode (For Jetson only)
 */
#define OSD_PROCESS_MODE 0

/* By default, OSD will not display text. To display text, change this to 1 */
#define OSD_DISPLAY_TEXT 1 // I changed to 0 to avoid displaying text
#define OSD_DISPLAY_BBOX 1 

/* Action recognition config */
static NvDsARConfig gActionConfig;

template <class T>
using Vec1D = std::vector<T>;

template <class T>
using Vec2D = std::vector<Vec1D<T>>;

template <class T>
using Vec3D = std::vector<Vec2D<T>>;


//gRPC part
using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using detector::BatchDetectionResults;
using detector::DetectionResults;
using detector::Detector;
using detector::ActionResult;
using detector::ActionClass;
using detector::Detection;
using detector::Point;


gint frame_number = 0;


// for fps computation
typedef struct {
    /** identifies the stream ID */
    guint32 stream_index;
    gdouble fps[MAX_STREAMS];
    gdouble fps_avg[MAX_STREAMS];
    guint32 num_instances;
    guint header_print_cnt;
    GMutex fps_lock;
    gpointer context;

    /** Test specific info */
    guint32 set_batch_size;
} PerfCtx;

typedef struct {
    GMutex *lock;
    int num_sources;
} LatencyCtx;



// for results from action classification model
std::vector<std::string> actions = {"bend", "sit", "sit_clap", "sit_phonecall", "sit_point", "sit_checktime", "sit_wave", "stand", "stand_clap", "stand_phonecall", "stand_point", "stand_checktime", "stand_wave", "walk", "fall", "unstable"};
std::unordered_map<int, int> objectClasses; // object_id, class_id

// grpc part

class DetectorClient {
 public:
  DetectorClient(std::shared_ptr<Channel> channel)
      : stub_(Detector::NewStub(channel)) {}

  ActionResult Detect(const BatchDetectionResults& batch) {
    ActionResult response;
    ClientContext context;

    Status status = stub_->Detect(&context, batch, &response);

    if (status.ok()) {
      std::cout << "Detection successful." << std::endl;
    } else {
      std::cout << "Detection failed with error code: " << status.error_code()
                << " and message: " << status.error_message() << std::endl;
    }

    return response;
  }
  
  // not needed
  ActionResult DetectTest(const BatchDetectionResults& batch) {
    ActionResult response;
    ClientContext context;

    Status status = stub_->DetectTest(&context, batch, &response);

    if (status.ok()) {
      std::cout << "Test detection successful." << std::endl;
    } else {
      std::cout << "Test detection failed with error code: " << status.error_code()
                << " and message: " << status.error_message() << std::endl;
    }

    return response;
  }

 private:
  std::unique_ptr<Detector::Stub> stub_;
};

// address of classification service
std::string target_str = "classification_app:8000";




/**
 * callback function to print the performance numbers of each stream.
*/
static void
perf_cb(gpointer context, NvDsAppPerfStruct *str) {
  PerfCtx *thCtx = (PerfCtx *) context;

  g_mutex_lock(&thCtx->fps_lock);
  /** str->num_instances is == num_sources */
  guint32 numf = str->num_instances;
  guint32 i;

  for (i = 0; i < numf; i++) {
    thCtx->fps[i] = str->fps[i];
    thCtx->fps_avg[i] = str->fps_avg[i];
  }
  thCtx->context = thCtx;
  g_print("**PERF: ");
  for (i = 0; i < numf; i++) {
    g_print("For source %d %.2f (%.2f)\t",i, thCtx->fps[i], thCtx->fps_avg[i]);
  }
  g_print("\n");
  g_mutex_unlock(&thCtx->fps_lock);
}
  
/**
 * callback function to print the latency of each component in the pipeline.
 */

static GstPadProbeReturn
latency_measurement_buf_prob(GstPad *pad, GstPadProbeInfo *info, gpointer u_data) {
  LatencyCtx *ctx = (LatencyCtx *) u_data;
  static int batch_num = 0;
  guint i = 0, num_sources_in_batch = 0;
  if (nvds_enable_latency_measurement) {
    GstBuffer *buf = (GstBuffer *) info->data;
    NvDsFrameLatencyInfo *latency_info = NULL;
    g_mutex_lock(ctx->lock);
    latency_info = (NvDsFrameLatencyInfo *)
        calloc(1, ctx->num_sources * sizeof(NvDsFrameLatencyInfo));;
    g_print("\n************BATCH-NUM = %d**************\n", batch_num);
    num_sources_in_batch = nvds_measure_buffer_latency(buf, latency_info);

    for (i = 0; i < num_sources_in_batch; i++) {
      g_print("Source id = %d Frame_num = %d Frame latency = %lf (ms) \n",
              latency_info[i].source_id,
              latency_info[i].frame_num,
              latency_info[i].latency);
    }
    g_mutex_unlock(ctx->lock);
    batch_num++;
  }

  return GST_PAD_PROBE_OK;
}



static void
cb_newpad(GstElement *decodebin, GstPad *decoder_src_pad, gpointer data)
{
  g_print("In cb_newpad\n");
  GstCaps *caps = gst_pad_get_current_caps(decoder_src_pad);
  const GstStructure *str = gst_caps_get_structure(caps, 0);
  const gchar *name = gst_structure_get_name(str);
  GstElement *source_bin = (GstElement *)data;
  GstCapsFeatures *features = gst_caps_get_features(caps, 0);

  /* Need to check if the pad created by the decodebin is for video and not
   * audio. */
  if (!strncmp(name, "video", 5))
  {
    /* Link the decodebin pad only if decodebin has picked nvidia
     * decoder plugin nvdec_*. We do this by checking if the pad caps contain
     * NVMM memory features. */
    if (gst_caps_features_contains(features, GST_CAPS_FEATURES_NVMM))
    {
      /* Get the source bin ghost pad */
      GstPad *bin_ghost_pad = gst_element_get_static_pad(source_bin, "src");
      if (!gst_ghost_pad_set_target(GST_GHOST_PAD(bin_ghost_pad),
                                    decoder_src_pad))
      {
        g_printerr("Failed to link decoder src pad to source bin ghost pad\n");
      }
      gst_object_unref(bin_ghost_pad);
    }
    else
    {
      g_printerr("Error: Decodebin did not pick nvidia decoder plugin.\n");
    }
  }
}


static void
decodebin_child_added(GstChildProxy *child_proxy, GObject *object,
                      gchar *name, gpointer user_data)
{
  g_print("Decodebin child added: %s\n", name);
  if (g_strrstr(name, "decodebin") == name)
  {
    g_signal_connect(G_OBJECT(object), "child-added",
                     G_CALLBACK(decodebin_child_added), user_data);
  }
}

static GstElement *
create_source_bin(guint index, const gchar *uri)
{
  GstElement *bin = NULL, *uri_decode_bin = NULL;
  gchar bin_name[16] = {};

  g_snprintf(bin_name, 15, "source-bin-%02d", index);
  /* Create a source GstBin to abstract this bin's content from the rest of the
   * pipeline */
  bin = gst_bin_new(bin_name);

  /* Source element for reading from the uri.
   * We will use decodebin and let it figure out the container format of the
   * stream and the codec and plug the appropriate demux and decode plugins. */
  uri_decode_bin = gst_element_factory_make("uridecodebin", "uri-decode-bin");

  if (!bin || !uri_decode_bin)
  {
    g_printerr("One element in source bin could not be created.\n");
    return NULL;
  }

  /* We set the input uri to the source element */
  g_object_set(G_OBJECT(uri_decode_bin), "uri", uri, NULL);

  /* Connect to the "pad-added" signal of the decodebin which generates a
   * callback once a new pad for raw data has beed created by the decodebin */
  g_signal_connect(G_OBJECT(uri_decode_bin), "pad-added",
                   G_CALLBACK(cb_newpad), bin);
  g_signal_connect(G_OBJECT(uri_decode_bin), "child-added",
                   G_CALLBACK(decodebin_child_added), bin);

  gst_bin_add(GST_BIN(bin), uri_decode_bin);

  /* We need to create a ghost pad for the source bin which will act as a proxy
   * for the video decoder src pad. The ghost pad will not have a target right
   * now. Once the decode bin creates the video decoder and generates the
   * cb_newpad callback, we will set the ghost pad target to the video decoder
   * src pad. */
  if (!gst_element_add_pad(bin, gst_ghost_pad_new_no_target("src",
                                                            GST_PAD_SRC)))
  {
    g_printerr("Failed to add ghost pad in source bin\n");
    return NULL;
  }

  return bin;
}


/*Method to parse information returned from the model*/
Vec2D<float>
parse_objects_from_tensor_meta(NvDsInferTensorMeta *tensor_meta)
{
  Vec1D<int> counts;
  Vec2D<float> peaks;
  float threshold = 0.15;
  void *cmap_data = tensor_meta->out_buf_ptrs_host[0];
  NvDsInferDims &cmap_dims = tensor_meta->output_layers_info[0].inferDims;
  /* Finding peaks within a given window */
  find_peak(peaks, cmap_data, cmap_dims, threshold);
  return peaks;
}


// (total keypoints) 18 * 2 (x,y) = 36 
int USER_ARRAY_SIZE = 36;

/* copy function set by user. "data" holds a pointer to NvDsUserMeta*/
static gpointer copy_user_meta(gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
  // gchar *src_user_metadata = (gchar*)user_meta->user_meta_data;
  // gchar *dst_user_metadata = (gchar*)g_malloc0(USER_ARRAY_SIZE);
  // memcpy(dst_user_metadata, src_user_metadata, USER_ARRAY_SIZE);
  guint *src_user_metadata = (guint*)user_meta->user_meta_data;
  guint *dst_user_metadata = (guint*)g_malloc0(USER_ARRAY_SIZE * sizeof(guint));
  // g_print("test");
  memcpy(dst_user_metadata, src_user_metadata, USER_ARRAY_SIZE * sizeof(guint));
  return (gpointer)dst_user_metadata;
}

/* release function set by user. "data" holds a pointer to NvDsUserMeta*/
static void release_user_meta(gpointer data, gpointer user_data)
{
  // g_print("Inside release user meta");
  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  if(user_meta->user_meta_data) {
    g_free(user_meta->user_meta_data);
    user_meta->user_meta_data = NULL;
  }
}
// end- to store user metadata inside object

/* MetaData to handle drawing onto the on-screen-display */
Vec1D<int>
create_display_meta(Vec2D<float> &normalized_peaks, NvDsFrameMeta *frame_meta, NvDsObjectMeta *obj_meta)
{
  float left = obj_meta->rect_params.left;
  float top = obj_meta->rect_params.top;
  float width = obj_meta->rect_params.width;
  float height = obj_meta->rect_params.height;

  int K = topology.size(); // number of limbs
  NvDsBatchMeta *bmeta = frame_meta->base_meta.batch_meta;
  NvDsDisplayMeta *dmeta = nvds_acquire_display_meta_from_pool(bmeta);

  NvDsMetaType user_meta_type = NVDS_USER_META; 

  NvDsMetaList *l_obj = NULL;
  nvds_add_display_meta_to_frame(frame_meta, dmeta);
  // g_print("source frame widht %d and source frame height %d \t", frame_meta->pipeline_width, frame_meta->pipeline_height);

  int C = normalized_peaks.size(); // number of joints detected in an object(here-human)
  int countJoints = 0; // to count number of joints detected, based on which we can filter out some poses??
  int XY_SKELETON = C * 2; // 17 * 2
  
  Vec1D<int> posePoints;
  for (int j = 0; j < XY_SKELETON; j++)
  {
    posePoints.push_back(0);
  }

  for (int j = 0; j < C; j++)
  {
    float k = normalized_peaks[j][0];
    if (k > 0)
    {
      auto &peak = normalized_peaks[j];
      float x = peak[1] * width + left;
      float y = peak[0] * height + top;
      if (dmeta->num_circles == MAX_ELEMENTS_IN_DISPLAY_META)
      {
        dmeta = nvds_acquire_display_meta_from_pool(bmeta);
        nvds_add_display_meta_to_frame(frame_meta, dmeta);
      }
      NvOSD_CircleParams &cparams = dmeta->circle_params[dmeta->num_circles];
      cparams.xc = x;
      cparams.yc = y;
      cparams.radius = 4;
      cparams.circle_color = NvOSD_ColorParams{244, 67, 54, 1};
      cparams.has_bg_color = 0;
      cparams.bg_color = NvOSD_ColorParams{0, 255, 0, 1};
      dmeta->num_circles++;

      posePoints[2*j] = x;
      posePoints[2*j+1] = y;
    }
    else
    {
      countJoints = countJoints + k; // k=-1 if no proper joint is detected
    }
  }

  for (int k = 0; k < K; k++) // for each limb
  {
    int c_a = topology[k][2];
    int c_b = topology[k][3];
    float k1 = normalized_peaks[c_a][0];
    float k2 = normalized_peaks[c_b][0];
    if (k1 > 0 && k2 > 0)
    {
      auto &peak0 = normalized_peaks[c_a];
      auto &peak1 = normalized_peaks[c_b];
      int x0 = peak0[1] * width + left;
      int y0 = peak0[0] * height + top;
      int x1 = peak1[1] * width + left;
      int y1 = peak1[0] * height + top;
      if (dmeta->num_lines == MAX_ELEMENTS_IN_DISPLAY_META)
      {
        dmeta = nvds_acquire_display_meta_from_pool(bmeta);
        nvds_add_display_meta_to_frame(frame_meta, dmeta);
      }
      NvOSD_LineParams &lparams = dmeta->line_params[dmeta->num_lines];
      lparams.x1 = x0;
      lparams.x2 = x1;
      lparams.y1 = y0;
      lparams.y2 = y1;
      lparams.line_width = 3;
      lparams.line_color = NvOSD_ColorParams{0, 255, 0, 1};
      dmeta->num_lines++;
    }
  }
  return posePoints;
}



// nvtracker_src_pad_src_pad_buffer_probe will extract metadata
// received from nvtracker
static GstPadProbeReturn
nvtracker_src_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info,
                          gpointer u_data)
{
  gchar *msg = NULL;
  GstBuffer *buf = (GstBuffer *)info->data;
  NvDsMetaList *l_frame = NULL;
  NvDsMetaList *l_obj = NULL;
  NvDsMetaList *l_user = NULL;
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
       l_frame = l_frame->next)
  {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
    
    int count = 0;

    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
         l_obj = l_obj->next)
    {
      // g_print("print something");
      NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)l_obj->data;
      guint id = obj_meta->object_id;
      g_print("id: %d \t", id);
      // g_print("tracked bounding box: %f", obj_meta->tracker_bbox_info.org_bbox_coords.width);
      count++;
    }
    g_print("count: %d \n", count);
  }
  return GST_PAD_PROBE_OK;
}


/* sgie_src_pad_buffer_probe  will extract metadata received from sgie
 * and extract poses, send to the classification service running separately */
static GstPadProbeReturn
sgie_src_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info,
                          gpointer u_data)
{
  // grpc
  std::string address(target_str);
  DetectorClient detector(grpc::CreateChannel(
      address, grpc::InsecureChannelCredentials()));

  gchar *msg = NULL;
  GstBuffer *buf = (GstBuffer *)info->data;
  NvDsMetaList *l_frame = NULL;
  NvDsMetaList *l_obj = NULL;
  NvDsMetaList *l_user = NULL;
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

  // gRPC message
  BatchDetectionResults batch_detection_results;

  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
       l_frame = l_frame->next)
  {
    // gRPC message
    DetectionResults* detection_results = batch_detection_results.add_detectionresult();

    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
    detection_results->set_sourceid(frame_meta->source_id);
    detection_results->set_framenum(frame_meta->frame_num);
    // frame_meta->bInferDone = 1;
    // g_print("Frame infer done status: %d", frame_meta->bInferDone); // equals 
    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next)
    {
      NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)(l_obj->data);
      guint id = obj_meta->object_id; //object id
      // gRPC message
      Detection* detection = detection_results->add_detection();
      detection->set_objectid(id);

      for (l_user = obj_meta->obj_user_meta_list; l_user != NULL;
         l_user = l_user->next)
      {
        NvDsUserMeta *user_meta = (NvDsUserMeta *)l_user->data;
        if (user_meta->base_meta.meta_type == NVDSINFER_TENSOR_OUTPUT_META)
        {
          NvDsInferTensorMeta *tensor_meta =
              (NvDsInferTensorMeta *)user_meta->user_meta_data;
          Vec2D<float> normalized_peak;
          normalized_peak = parse_objects_from_tensor_meta(tensor_meta);

          Vec1D<int> posePoints;
          posePoints = create_display_meta(normalized_peak, frame_meta, obj_meta);
          
          int C = normalized_peak.size();
          
          // int XY_SKELETON = C * 2; 
          for (int j=0; j<C; j++)
          {
            // gRPC message
            Point* point = detection->add_points();
            point->set_x(posePoints[2*j]);
            point->set_y(posePoints[2*j+1]);
            // g_print("\n------------%d   %d----------", posePoints[2*j], posePoints[2*j+1]);

          }
        }
      }
    }
  }

  g_print("\n Calling gRPC... \n");

  ActionResult result = detector.Detect(batch_detection_results);
  for (const auto& action_class : result.classes()) {
     std::cout << "Object ID: " << action_class.objectid() << ", Class: " << action_class.classid() << std::endl;
  }

  // putting label in the object metadata
  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
       l_frame = l_frame->next)
  {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);

    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
         l_obj = l_obj->next)
    {
      NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)l_obj->data;
      guint id = obj_meta->object_id; //object id
      g_print("%d ", id);
      for (const auto& action_class : result.classes()) {
        // std::cout << "Object ID: " << action_class.objectid() << ", Class: " << action_class.classid() << std::endl;
        if (id == action_class.objectid()) {
          NvOSD_TextParams *txt_params = &obj_meta->text_params;
          txt_params->display_text = (char *)g_malloc0(MAX_DISPLAY_LEN);
          guint actionClassID = action_class.classid();
          objectClasses[id] = actionClassID;
          snprintf(txt_params->display_text, MAX_DISPLAY_LEN, "%s: %d", actions[actionClassID].c_str(), id);
          //txt_params->display_text = (char *) "class_name";
          obj_meta->class_id = actionClassID;
        }
      }
      bool found = false;
      // auto it = std::find(objectClasses.begin(), objectClasses.end(), id);
      for (auto const& classInfo : objectClasses) {
        if (classInfo.first == id) {
            found = true;
            break;
        }
      }
      if (found)
      {
        // id is present in the vector
        guint actionClassID = objectClasses[id];
        NvOSD_TextParams *txt_params = &obj_meta->text_params;
        txt_params->display_text = (char *)g_malloc0(MAX_DISPLAY_LEN);
        snprintf(txt_params->display_text, MAX_DISPLAY_LEN, "%s: %d", actions[actionClassID].c_str(), id);
        obj_meta->class_id = actionClassID;
      } 
    }
  }

  return GST_PAD_PROBE_OK;
}

/* osd_sink_pad_buffer_probe  will extract metadata received from OSD
 * and update params for drawing rectangle, object information etc. */
static GstPadProbeReturn
osd_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info,
                          gpointer u_data)
{
  GstBuffer *buf = (GstBuffer *)info->data;
  guint num_rects = 0;
  NvDsObjectMeta *obj_meta = NULL;
  NvDsMetaList *l_frame = NULL;
  NvDsMetaList *l_obj = NULL;
  NvDsDisplayMeta *display_meta = NULL;

  NvDsMetaList *l_user = NULL;

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
       l_frame = l_frame->next)
  {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
    display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
    nvds_add_display_meta_to_frame(frame_meta, display_meta);


    // testing bounding box display extracting from object metadata
    // g_print("entering inside to display");
    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next)
    {
      // g_print("entered inside to display");
      NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)l_obj->data;
      // NvBbox_Coords bboxcoords = obj_meta->detector_bbox_info.org_bbox_coords;

      // drawing rectangle
      NvOSD_RectParams *rparams = &display_meta->rect_params[display_meta->num_rects];
      rparams->left = obj_meta->rect_params.left; // float in pixels
      rparams->top = obj_meta->rect_params.top;
      rparams->width = obj_meta->rect_params.width;
      rparams->height = obj_meta->rect_params.height;
      rparams->border_width = 4;
      rparams->border_color = NvOSD_ColorParams{255, 255, 0, 1};
      rparams->has_bg_color = false;
      display_meta->num_rects++;
      // nvds_add_display_meta_to_frame(frame_meta, dmeta); // just adding this caused some effect while displaying estimated pose, so add like below
      if (display_meta->num_rects == MAX_ELEMENTS_IN_DISPLAY_META)
      {
        display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
        nvds_add_display_meta_to_frame(frame_meta, display_meta);
      }

      // g_print("near obj_user_metadata loop \t");
      // getting inside usermeta of the object
      for (l_user = obj_meta->obj_user_meta_list; l_user != NULL;
           l_user = l_user->next)
      {
        // g_print("inside obj meta data of user");
        NvDsUserMeta *user_meta = (NvDsUserMeta *)l_user->data;
        NvDsInferTensorMeta *tensor_meta = (NvDsInferTensorMeta *)user_meta->user_meta_data;
        guint *pose_information = (guint*)user_meta->user_meta_data;
        // g_print("Pose information2: %d, %d \n", pose_information[10], pose_information[11]);
      }
    }


    int offset = 0;
    /* Parameters to draw text onto the On-Screen-Display */
    NvOSD_TextParams *txt_params = &display_meta->text_params[0];
    display_meta->num_labels = 1;
    txt_params->display_text = (char *)g_malloc0(MAX_DISPLAY_LEN);
    offset = snprintf(txt_params->display_text, MAX_DISPLAY_LEN, "Frame Number =  %d", frame_number);
    offset = snprintf(txt_params->display_text + offset, MAX_DISPLAY_LEN, " ");

    txt_params->x_offset = 10;
    txt_params->y_offset = 12;

    txt_params->font_params.font_name = (char *)"Mono";
    txt_params->font_params.font_size = 10;
    txt_params->font_params.font_color.red = 1.0;
    txt_params->font_params.font_color.green = 1.0;
    txt_params->font_params.font_color.blue = 1.0;
    txt_params->font_params.font_color.alpha = 1.0;

    txt_params->set_bg_clr = 1;
    txt_params->text_bg_clr.red = 0.0;
    txt_params->text_bg_clr.green = 0.0;
    txt_params->text_bg_clr.blue = 0.0;
    txt_params->text_bg_clr.alpha = 1.0;

  }
  frame_number++;
  return GST_PAD_PROBE_OK;
}


static gboolean
bus_call(GstBus *bus, GstMessage *msg, gpointer data)
{
  GMainLoop *loop = (GMainLoop *)data;
  switch (GST_MESSAGE_TYPE(msg))
  {
  case GST_MESSAGE_EOS:
    g_print("End of Stream\n");
    g_main_loop_quit(loop);
    break;

  case GST_MESSAGE_ERROR:
  {
    gchar *debug;
    GError *error;
    gst_message_parse_error(msg, &error, &debug);
    g_printerr("ERROR from element %s: %s\n",
               GST_OBJECT_NAME(msg->src), error->message);
    if (debug)
      g_printerr("Error details: %s\n", debug);
    g_free(debug);
    g_error_free(error);
    g_main_loop_quit(loop);
    break;
  }

  default:
    break;
  }
  return TRUE;
}

gboolean
link_element_to_tee_src_pad(GstElement *tee, GstElement *sinkelem)
{
  gboolean ret = FALSE;
  GstPad *tee_src_pad = NULL;
  GstPad *sinkpad = NULL;
  GstPadTemplate *padtemplate = NULL;

  padtemplate = (GstPadTemplate *)gst_element_class_get_pad_template(GST_ELEMENT_GET_CLASS(tee), "src_%u");
  tee_src_pad = gst_element_request_pad(tee, padtemplate, NULL, NULL);

  if (!tee_src_pad)
  {
    g_printerr("Failed to get src pad from tee");
    goto done;
  }

  sinkpad = gst_element_get_static_pad(sinkelem, "sink");
  if (!sinkpad)
  {
    g_printerr("Failed to get sink pad from '%s'",
               GST_ELEMENT_NAME(sinkelem));
    goto done;
  }

  if (gst_pad_link(tee_src_pad, sinkpad) != GST_PAD_LINK_OK)
  {
    g_printerr("Failed to link '%s' and '%s'", GST_ELEMENT_NAME(tee),
               GST_ELEMENT_NAME(sinkelem));
    goto done;
  }
  ret = TRUE;

done:
  if (tee_src_pad)
  {
    gst_object_unref(tee_src_pad);
  }
  if (sinkpad)
  {
    gst_object_unref(sinkpad);
  }
  return ret;
}


// tracker config file reader and setting tracker properties
/* Tracker config parsing */

#define CHECK_ERROR(error) \
    if (error) { \
        g_printerr ("Error while parsing config file: %s\n", error->message); \
        goto done; \
    }

#define CONFIG_GROUP_TRACKER "tracker"
#define CONFIG_GROUP_TRACKER_WIDTH "tracker-width"
#define CONFIG_GROUP_TRACKER_HEIGHT "tracker-height"
#define CONFIG_GROUP_TRACKER_LL_CONFIG_FILE "ll-config-file"
#define CONFIG_GROUP_TRACKER_LL_LIB_FILE "ll-lib-file"
#define CONFIG_GROUP_TRACKER_ENABLE_BATCH_PROCESS "enable-batch-process"
#define CONFIG_GPU_ID "gpu-id"
#define CONFIG_TRACK_ID_DISPLAY "display-tracking-id"

static gchar *
get_absolute_file_path (gchar *cfg_file_path, gchar *file_path)
{
  gchar abs_cfg_path[PATH_MAX + 1];
  gchar *abs_file_path;
  gchar *delim;

  if (file_path && file_path[0] == '/') {
    return file_path;
  }

  if (!realpath (cfg_file_path, abs_cfg_path)) {
    g_free (file_path);
    return NULL;
  }

  // Return absolute path of config file if file_path is NULL.
  if (!file_path) {
    abs_file_path = g_strdup (abs_cfg_path);
    return abs_file_path;
  }

  delim = g_strrstr (abs_cfg_path, "/");
  *(delim + 1) = '\0';

  abs_file_path = g_strconcat (abs_cfg_path, file_path, NULL);
  g_free (file_path);

  return abs_file_path;
}

static gboolean
set_tracker_properties (GstElement *nvtracker)
{
  gboolean ret = FALSE;
  GError *error = NULL;
  gchar **keys = NULL;
  gchar **key = NULL;
  GKeyFile *key_file = g_key_file_new ();

  if (!g_key_file_load_from_file (key_file, TRACKER_CONFIG_FILE, G_KEY_FILE_NONE,
          &error)) {
    g_printerr ("Failed to load config file: %s\n", error->message);
    return FALSE;
  }

  keys = g_key_file_get_keys (key_file, CONFIG_GROUP_TRACKER, NULL, &error);
  CHECK_ERROR (error);

  for (key = keys; *key; key++) {
    if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_WIDTH)) {
      gint width =
          g_key_file_get_integer (key_file, CONFIG_GROUP_TRACKER,
          CONFIG_GROUP_TRACKER_WIDTH, &error);
      CHECK_ERROR (error);
      g_object_set (G_OBJECT (nvtracker), "tracker-width", width, NULL);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_HEIGHT)) {
      gint height =
          g_key_file_get_integer (key_file, CONFIG_GROUP_TRACKER,
          CONFIG_GROUP_TRACKER_HEIGHT, &error);
      CHECK_ERROR (error);
      g_object_set (G_OBJECT (nvtracker), "tracker-height", height, NULL);
    
    } else if (!g_strcmp0 (*key, CONFIG_TRACK_ID_DISPLAY)) {
      gboolean track_id_display =
          g_key_file_get_integer (key_file, CONFIG_GROUP_TRACKER,
          CONFIG_TRACK_ID_DISPLAY, &error);
      CHECK_ERROR (error);
      g_object_set (G_OBJECT (nvtracker), "display-tracking-id", track_id_display, NULL);


    } else if (!g_strcmp0 (*key, CONFIG_GPU_ID)) {
      guint gpu_id =
          g_key_file_get_integer (key_file, CONFIG_GROUP_TRACKER,
          CONFIG_GPU_ID, &error);
      CHECK_ERROR (error);
      g_object_set (G_OBJECT (nvtracker), "gpu_id", gpu_id, NULL);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_LL_CONFIG_FILE)) {
      char* ll_config_file = get_absolute_file_path ((char *) TRACKER_CONFIG_FILE,
                g_key_file_get_string (key_file,
                    CONFIG_GROUP_TRACKER,
                    CONFIG_GROUP_TRACKER_LL_CONFIG_FILE, &error));
      CHECK_ERROR (error);
      g_object_set (G_OBJECT (nvtracker), "ll-config-file", ll_config_file, NULL);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_LL_LIB_FILE)) {
      char* ll_lib_file = get_absolute_file_path ((char *) TRACKER_CONFIG_FILE,
                g_key_file_get_string (key_file,
                    CONFIG_GROUP_TRACKER,
                    CONFIG_GROUP_TRACKER_LL_LIB_FILE, &error));
      CHECK_ERROR (error);
      g_object_set (G_OBJECT (nvtracker), "ll-lib-file", ll_lib_file, NULL);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_ENABLE_BATCH_PROCESS)) {
      gboolean enable_batch_process =
          g_key_file_get_integer (key_file, CONFIG_GROUP_TRACKER,
          CONFIG_GROUP_TRACKER_ENABLE_BATCH_PROCESS, &error);
      CHECK_ERROR (error);
      g_object_set (G_OBJECT (nvtracker), "enable_batch_process",
                    enable_batch_process, NULL);
    } else {
      g_printerr ("Unknown key '%s' for group [%s]", *key,
          CONFIG_GROUP_TRACKER);
    }
  }

  ret = TRUE;
done:
  if (error) {
    g_error_free (error);
  }
  if (keys) {
    g_strfreev (keys);
  }
  if (!ret) {
    g_printerr ("%s failed", __func__);
  }
  return ret;
}


//////////////////////// MAIN START  ///////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{
  GMainLoop *loop = NULL;

  GstCaps *caps = NULL;
  GstElement *pipeline = NULL, *bishal_sink = NULL, *tiler = NULL, *streammux = NULL,
             *pgie = NULL, *sgie1 = NULL, *nvtracker = NULL, *nvvidconv = NULL, *nvosd = NULL, *filesink = NULL, *queue2 = NULL,
             *nvvideoconvert = NULL, *tee = NULL, *h264encoder = NULL, *cap_filter = NULL, *queue = NULL, *qtmux = NULL, *h264parser1 = NULL;

  GstBus *bus = NULL;
  guint bus_watch_id;
  GstPad *osd_sink_pad = NULL;
  
  guint tiler_rows, tiler_columns;
  guint num_sources;
  gchar* sink_type;
  guint pgie_batch_size;


  /* Add a transform element for Jetson*/
  #ifdef PLATFORM_TEGRA
    GstElement *transform = NULL;
  #endif

  
  /* Check input arguments */
  if (argc < 3 || strncmp(argv[1], "-c", 3))
  {
    g_printerr("Usage: %s -c <config.txt>\n", argv[0]);
    return -1;
  }

  if (!parse_action_config(argv[2], gActionConfig)) {
    g_printerr("parse config file: %s failed.\n", argv[2]);
    return -1;
  }
 
  // get num_source from the uri_list size
  num_sources = gActionConfig.uri_list.size();
  g_print("Number of source: %d", num_sources);


  /* Standard GStreamer initialization */
  gst_init(&argc, &argv);
  loop = g_main_loop_new(NULL, FALSE);

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new("deepstream-tensorrt-openpose-pipeline");

  /* Create nvstreammux instance to form batches from one or more sources. */
  streammux = gst_element_factory_make("nvstreammux", "stream-muxer");

  if (!pipeline || !streammux)
  {
    g_printerr("One element could not be created. Exiting.\n");
    return -1;
  }

  // primary inference engine - for yolo model
  pgie = gst_element_factory_make("nvinfer", "primary-nvinference-engine");
  
  // secondary inference engine - for vitpose model
  sgie1 = gst_element_factory_make ("nvinfer", "secondary1-nvinference-engine");

  // tracker
  nvtracker = gst_element_factory_make ("nvtracker", "tracker");

  tiler = gst_element_factory_make("nvmultistreamtiler", "nvtiler");

  /* Use convertor to convert from NV12 to RGBA as required by nvosd */
  nvvidconv = gst_element_factory_make("nvvideoconvert", "nvvideo-converter");


  /* Create OSD to draw on the converted RGBA buffer */
  nvosd = gst_element_factory_make("nvdsosd", "nv-onscreendisplay");

  // sink type from config.txt
  g_print("\nSink type is %s \n", gActionConfig.sink_type.c_str());
  bishal_sink = gst_element_factory_make(gActionConfig.sink_type.c_str(), "nvvideo-renderer");

   // filesink to save file
  filesink = gst_element_factory_make("filesink", "filesink");
  /* Set output file location */
  g_object_set(G_OBJECT(filesink), "location", "/opt/nvidia/deepstream/deepstream/sources/apps/Deepstream-app/deepstream_yolo_vitpose_grpc/Poseestimation.mp4", NULL);
  
  // other required elements in the pipeline
  queue = gst_element_factory_make("queue", "queue");
  queue2 = gst_element_factory_make("queue", "queue2");
  nvvideoconvert = gst_element_factory_make("nvvideoconvert", "nvvideo-converter1");
  
  // tee is needed to implement two types of sink- saving video file and displaying
  tee = gst_element_factory_make("tee", "TEE");

  h264encoder = gst_element_factory_make("nvv4l2h264enc", "video-encoder");
  cap_filter = gst_element_factory_make("capsfilter", "enc_caps_filter");
  caps = gst_caps_from_string("video/x-raw(memory:NVMM), format=I420");
  g_object_set(G_OBJECT(cap_filter), "caps", caps, NULL);
  h264parser1 = gst_element_factory_make("h264parse", "h264-parser1");
  qtmux = gst_element_factory_make("qtmux", "muxer");


  #ifdef PLATFORM_TEGRA
    transform = gst_element_factory_make("nvegltransform", "nvegl-transform");
  #endif

  #ifdef PLATFORM_TEGRA
    if (!transform)
    {
      g_printerr("One tegra element could not be created. Exiting.\n");
      return -1;
    }
  #endif

  if (!nvtracker || !nvvidconv || !nvosd || !bishal_sink || !tiler || !sgie1 || !queue || !queue2 || !tee || !nvvideoconvert || !cap_filter || !h264encoder || !h264parser1 || !qtmux || !filesink)
  {
    g_printerr("One element could not be created check check. Exiting.\n");
    return -1;
  }


  g_object_set(G_OBJECT(streammux), "batch-size", num_sources, NULL);
  g_object_set(G_OBJECT(streammux), "width", gActionConfig.muxer_width, "height",
               gActionConfig.muxer_height, "batch_size", num_sources,
               "batched-push-timeout", gActionConfig.muxer_batch_timeout, NULL);

  /* Set all the necessary properties of the nvinfer element,
   * the necessary ones are : */
  g_object_set (G_OBJECT (pgie), "config-file-path", "/opt/nvidia/deepstream/deepstream/sources/apps/Deepstream-app/deepstream_yolo_vitpose_grpc/dstest2_pgie_config.txt", NULL);
  g_object_set(G_OBJECT(sgie1), "output-tensor-meta", TRUE,
               "config-file-path", "/opt/nvidia/deepstream/deepstream/sources/apps/Deepstream-app/deepstream_yolo_vitpose_grpc/dstest2_sgie1_config.txt", NULL);


  /* Override the batch-size set in the config file with the number of sources. */
  g_object_get (G_OBJECT (pgie), "batch-size", &pgie_batch_size, NULL);
  if (pgie_batch_size != num_sources) {
    g_printerr
        ("WARNING: Overriding infer-config batch-size (%d) with number of sources (%d)\n",
        pgie_batch_size, num_sources);
    g_object_set (G_OBJECT (pgie), "batch-size", num_sources, NULL);
  }

  /* Set necessary properties of the tracker element. */
  if (!set_tracker_properties(nvtracker)) {
    g_printerr ("Failed to set tracker properties. Exiting.\n");
    return -1;
  }

  /* we set the tiler properties here */
  tiler_rows = (guint)sqrt(num_sources);
  tiler_columns = (guint)ceil(1.0 * num_sources / tiler_rows);
  g_object_set(G_OBJECT(tiler), "rows", tiler_rows, "columns", tiler_columns,
                "width", gActionConfig.tiler_width, "height", gActionConfig.tiler_height, NULL);

  g_object_set(G_OBJECT(nvosd), "process-mode", OSD_PROCESS_MODE,
                "display-text", OSD_DISPLAY_TEXT, "display-bbox", OSD_DISPLAY_BBOX, NULL);

  g_object_set(G_OBJECT(bishal_sink), "sync", gActionConfig.display_sync, NULL);



  /* we add a message handler */
  bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
  bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
  gst_object_unref(bus);


  // adding streammux to the pipeline
  gst_bin_add(GST_BIN(pipeline), streammux);
  
  // multiple stream source to streammux
  for (int i = 0; i < num_sources; i++)
  {
    GstPad *sinkpad, *srcpad;
    gchar pad_name[16] = {};
    GstElement *source_bin = create_source_bin(i, gActionConfig.uri_list[i].c_str());

    if (!source_bin)
    {
      g_printerr("Failed to create source bin. Exiting.\n");
      return -1;
    }
    
    gst_bin_add(GST_BIN(pipeline), source_bin);

    g_snprintf(pad_name, 15, "sink_%u", i);
    sinkpad = gst_element_get_request_pad(streammux, pad_name);
    if (!sinkpad)
    {
      g_printerr("Streammux request sink pad failed. Exiting.\n");
      return -1;
    }

    srcpad = gst_element_get_static_pad(source_bin, "src");
    if (!srcpad)
    {
      g_printerr("Failed to get src pad of source bin. Exiting.\n");
      return -1;
    }

    if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK)
    {
      g_printerr("Failed to link source bin to stream muxer. Exiting.\n");
      return -1;
    }

    gst_object_unref(srcpad);
    gst_object_unref(sinkpad);
  }


  /* Set up the pipeline */
  /* we add all elements into the pipeline */
  #ifdef PLATFORM_TEGRA
    gst_bin_add_many(GST_BIN(pipeline), transform, pgie, nvtracker, sgie1, tiler, nvvidconv, nvosd, bishal_sink, tee, nvvideoconvert, h264encoder, cap_filter, queue, queue2, h264parser1, qtmux, filesink NULL);
  #else
    gst_bin_add_many(GST_BIN(pipeline), pgie, nvtracker, sgie1, tiler, nvvidconv, nvosd, bishal_sink, tee, nvvideoconvert, h264encoder, cap_filter, queue, queue2, h264parser1, qtmux, filesink, NULL);
  #endif


  #if 0
    #ifdef PLATFORM_TEGRA
      if (!gst_element_link_many (streammux, pgie,
              nvvidconv, nvosd, transform, sink, NULL)) {
        g_printerr ("Elements could not be linked: 2. Exiting.\n");
        return -1;
      }
    #else
      if (!gst_element_link_many (streammux, pgie, nvvidconv, nvosd, sink, NULL)) {
        g_printerr ("Elements could not be linked: 2. Exiting.\n");
        return -1;
      }
    #endif

  #else
    #ifdef PLATFORM_TEGRA
      if (!gst_element_link_many(,streammux, pgie, nvtracker, sgie1, tiler, nvvidconv, nvosd, tee NULL))
      {
        g_printerr("Elements could not be linked: 2. Exiting.\n");
        return -1;
      }
    #else
      if (!gst_element_link_many(streammux, pgie, nvtracker, sgie1, tiler, nvvidconv, nvosd, tee, NULL)) 
      {
        g_printerr("Elements could not be linked: 2. Exiting.\n");
        return -1;
      }
    #endif

    #if 0
      if (!link_element_to_tee_src_pad(tee, queue)) {
          g_printerr ("Could not link tee to sink\n");
          return -1;
      }
      if (!gst_element_link_many (queue, sink, NULL)) {
        g_printerr ("Elements could not be linked: 2. Exiting.\n");
        return -1;
      }
    #else
      if (!link_element_to_tee_src_pad(tee, queue))
      {
        g_printerr("Could not link tee to nvvideoconvert\n");
        return -1;
      }

      // to save output video file
      if (!gst_element_link_many(queue, nvvideoconvert, cap_filter, h264encoder,
                                h264parser1, qtmux, filesink, NULL))
      {
        g_printerr("Elements could not be linked\n");
        return -1;
      }

      if (!link_element_to_tee_src_pad(tee, queue2))
      {
        g_printerr("Could not link tee to nvvideoconvert\n");
        return -1;
      }

      // to display
      if (!gst_element_link_many(queue2, bishal_sink, NULL))
      {
        g_printerr("Elements could not be linked\n");
        return -1;
      }

    #endif
  #endif



  // for fps calculation
  GstPad *sink_pad = gst_element_get_static_pad(nvvidconv, "src");
  if (!sink_pad)
    g_print("Unable to get sink pad\n");
  else {
    LatencyCtx *ctx = (LatencyCtx *) g_malloc0(sizeof(LatencyCtx));
    ctx->lock = (GMutex *) g_malloc0(sizeof(GMutex));
    ctx->num_sources = num_sources;
    gst_pad_add_probe(sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
                      latency_measurement_buf_prob, ctx, NULL);
  }
  gst_object_unref(sink_pad);
  GstPad *conv_pad = gst_element_get_static_pad(nvvidconv, "sink");
  if (!conv_pad)
    g_print("Unable to get conv_pad pad\n");
  else {
    NvDsAppPerfStructInt *str = (NvDsAppPerfStructInt *) g_malloc0(sizeof(NvDsAppPerfStructInt));
    PerfCtx *perf_ctx = (PerfCtx *) g_malloc0(sizeof(PerfCtx));
    g_mutex_init(&perf_ctx->fps_lock);
    str->context = perf_ctx;
    str->num_instances = num_sources;
    enable_perf_measurement(str, conv_pad, num_sources, 1, 0, perf_cb);
  }
  gst_object_unref(conv_pad);



  // adding probe to process the output from the pose estimation model to get poses
  GstPad *sgie_src_pad = gst_element_get_static_pad(sgie1, "src");
  if (!sgie_src_pad)
    g_print("Unable to get sgie src pad\n");
  else
    gst_pad_add_probe(sgie_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
                      sgie_src_pad_buffer_probe, NULL, NULL);


 // adding probe to check the tracking information of bounding box
  // GstPad *nvtracker_src_pad = gst_element_get_static_pad(nvtracker, "src");
  // if (!nvtracker_src_pad)
  //   g_print("Unable to get nvtracker src pad\n");
  // else
  //   gst_pad_add_probe(nvtracker_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
  //                     nvtracker_src_pad_buffer_probe, NULL, NULL);


  /* Lets add probe to get informed of the meta data generated, we add probe to
   * the sink pad of the osd element, since by that time, the buffer would have
   * had got all the metadata. */
  // osd_sink_pad = gst_element_get_static_pad(nvosd, "sink");
  // if (!osd_sink_pad)
  //   g_print("Unable to get sink pad\n");
  // else
  //   gst_pad_add_probe(osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
  //                     osd_sink_pad_buffer_probe, NULL, NULL);



  /* Set the pipeline to "playing" state */
  g_print("Now playing: %s\n", argv[1]);
  gst_element_set_state(pipeline, GST_STATE_PLAYING);

  /* Wait till pipeline encounters an error or EOS */
  g_print("Running...\n");
  g_main_loop_run(loop);

  /* Out of the main loop, clean up nicely */
  g_print("Returned, stopping playback\n");
  gst_element_set_state(pipeline, GST_STATE_NULL);
  g_print("Deleting pipeline\n");
  gst_object_unref(GST_OBJECT(pipeline));
  g_source_remove(bus_watch_id);
  g_main_loop_unref(loop);
  return 0;
}