# -lcurand -lcublas -lcusparse -lcufft -lnpp -lcudart

get_device_properties : get_device_properties.cu
	nvcc  -I. -I/usr/local/cuda-5.0/include/  get_device_properties.cu -o get_device_properties


clean:
	rm -f *.o *~ core .depend get_device_properties

depend .depend dep:
	$(CC) $(CFLAGS) -M *.c > $@


ifeq (.depend,$(wildcard .depend))
include .depend
endif
