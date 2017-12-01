/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.utils.io

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.hadoop.mapreduce._
import org.apache.hadoop.mapreduce.lib.output.{FileOutputCommitter, FileOutputFormat}

class DirectMapreduceOutputCommitter extends OutputCommitter {

  override def setupJob(jobContext: JobContext): Unit = {}

  override def setupTask(taskContext: TaskAttemptContext): Unit = {}

  override def needsTaskCommit(taskContext: TaskAttemptContext): Boolean = {
    // We return true here to guard against implementations that do not handle false correctly.
    // The meaning of returning false is not entirely clear, so it's possible to be interpreted
    // as an error. Returning true just means that commitTask() will be called, which is a no-op.
    true
  }

  override def commitTask(taskContext: TaskAttemptContext): Unit = {}

  override def abortTask(taskContext: TaskAttemptContext): Unit = {}

  /**
   * Creates a _SUCCESS file to indicate the entire job was successful.
   * This mimics the behavior of FileOutputCommitter, reusing the same file name and conf option.
   */
  override def commitJob(context: JobContext): Unit = {
    val conf = context.getConfiguration
    if (shouldCreateSuccessFile(conf)) {
      val outputPath = FileOutputFormat.getOutputPath(context)
      if (outputPath != null) {
        val fileSys = outputPath.getFileSystem(conf)
        val filePath = new Path(outputPath, FileOutputCommitter.SUCCEEDED_FILE_NAME)
        fileSys.create(filePath).close()
      }
    }
  }

  /** By default, we do create the _SUCCESS file, but we allow it to be turned off. */
  private def shouldCreateSuccessFile(conf: Configuration): Boolean = {
    conf.getBoolean("mapreduce.fileoutputcommitter.marksuccessfuljobs", true)
  }
}
