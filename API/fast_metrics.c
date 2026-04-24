#define _GNU_SOURCE
#include <fcntl.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

// Function to fetch network bytes for a specific container PID
long long get_container_rx_bytes(int target_pid, const char* interface_name) {
    char net_ns_path[256];
    char stats_path[256];
    char buffer[256];
    long long rx_bytes = -1;

    // Format the path to the container's network namespace
    snprintf(net_ns_path, sizeof(net_ns_path), "/proc/%d/ns/net", target_pid);

    // Open the namespace file descriptor
    int fd = open(net_ns_path, O_RDONLY);
    if (fd == -1) {
        perror("Failed to open network namespace");
        return -1;
    }

    // Enter the namespace using setns()
    // CLONE_NEWNET ensures we are specifically joining a network namespace
    if (setns(fd, CLONE_NEWNET) == -1) {
        perror("Failed to setns");
        close(fd); // ALWAYS close on failure
        return -1;
    }

    // Close the file descriptor immediately to prevent leaks
    close(fd);


    // Read the metrics directly from the sysfs tree for the target interface (e.g., "eth0")
    snprintf(stats_path, sizeof(stats_path), "/sys/class/net/%s/statistics/rx_bytes", interface_name);
    
    FILE *f = fopen(stats_path, "r");
    if (f) {
        if (fgets(buffer, sizeof(buffer), f) != NULL) {
            rx_bytes = atoll(buffer);
        }
        fclose(f);
    }
    
    return rx_bytes;
}