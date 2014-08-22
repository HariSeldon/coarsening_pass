#ifndef QUEUE_H
#define QUEUE_H

#include <vector>

#include <CL/cl.h>

class Buffer;
class Context;
class Device;
class Event;
class Kernel;

class Queue {

public:
  enum CommandQueueFlags {
    OutOfOrder = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
    EnableProfiling = CL_QUEUE_PROFILING_ENABLE
  };

// Constructors and Destructors.
//------------------------------------------------------------------------------
public:
  Queue(const Context& context, const Device& device, 
        CommandQueueFlags properties);
  ~Queue() throw();

// Public methods.
//------------------------------------------------------------------------------
public:
  void readBuffer(const Buffer& buffer, size_t size, void* pointer);
  void readBuffer(const Buffer& buffer, size_t size, void* pointer, 
                  Event& event);
  void readBuffer(const Buffer& buffer, 
                  bool blocking, 
                  size_t offset, 
                  size_t size, 
                  void* pointer, 
                  const std::vector<Event*>& eventWaitList,
                  Event& event);

//------------------------------------------------------------------------------
  void writeBuffer(const Buffer& buffer, size_t size, void* pointer);
  void writeBuffer(const Buffer& buffer, size_t size, void* pointer, 
                   Event& event);
  void writeBuffer(const Buffer& buffer, 
                   bool blocking, 
                   size_t offset, 
                   size_t size, 
                   void* pointer, 
                   const std::vector<Event*>& eventWaitList,
                   Event& event);

//------------------------------------------------------------------------------
  void run(const Kernel& kernel, 
           unsigned int dimensionsNumber,
           const size_t *globalOffset,
           const size_t *globalSize,
           const size_t *localSize);
  void run(const Kernel& kernel, 
           unsigned int dimensionsNumber,
           const size_t *globalOffset,
           const size_t *globalSize,
           const size_t *localSize,
           Event& event);
  void run(const Kernel& kernel, 
           unsigned int dimensionsNumber,
           const size_t *globalOffset,
           const size_t *globalSize,
           const size_t *localSize,
           const std::vector<Event*>& eventWaitList,
           Event& event);

  void runTask(const Kernel& kernel);
  void runTask(const Kernel& kernel, Event& event);
  void runTask(const Kernel& kernel,
               const std::vector<Event*>& eventWaitList,
               Event& event);

  void finish();
  void flush();

// Private Fields.
//------------------------------------------------------------------------------
private:
  cl_command_queue queue;

// Private Methods.
//------------------------------------------------------------------------------
private:
  static inline void readBuffer(cl_command_queue queue,
                                cl_mem buffer,
                                cl_bool blocking,
                                size_t offset,
                                size_t size,
                                void* pointer,
                                cl_uint waitListSize,
                                const cl_event* waitList,
                                cl_event* event);  
  
  static inline void writeBuffer(cl_command_queue queue,
                                 cl_mem buffer, 
                                 cl_bool blocking, 
                                 size_t offset, 
                                 size_t size, 
                                 void* pointer, 
                                 cl_uint waitListSize, 
                                 const cl_event* waitList, 
                                 cl_event* event);

  static inline void run(cl_command_queue queue,
                         cl_kernel kernel,
                         unsigned int dimensionsNumber,
                         const size_t *globalOffset,
                         const size_t *globalSize,
                         const size_t *localSize,
                         cl_uint waitListSize,
                         const cl_event* waitList,
                         cl_event* event);

  static inline void runTask(cl_command_queue queue,
                             cl_kernel kernel,
                             cl_uint waitListSize,
                             const cl_event* waitList,
                             cl_event* event);

  inline void getEventsId(const std::vector<Event*>& eventWaitList,
                          cl_event* eventsId) const;
};

#endif
