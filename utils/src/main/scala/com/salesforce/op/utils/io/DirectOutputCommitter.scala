// scalastyle:off header.matches
/*
 * Modifications: (c) 2017, Salesforce.com, Inc.
 * Copyright 2015 Databricks, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License.  You may obtain
 * a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


package com.salesforce.op.utils.io

import org.apache.hadoop.fs.Path
import org.apache.hadoop.mapred._

/**
 * OutputCommitter suitable for S3 workloads. Unlike the usual FileOutputCommitter, which
 * writes files to a _temporary/ directory before renaming them to their final location, this
 * simply writes directly to the final location.
 *
 * The FileOutputCommitter is required for HDFS + speculation, which allows only one writer at
 * a time for a file (so two people racing to write the same file would not work). However, S3
 * supports multiple writers outputting to the same file, where visibility is guaranteed to be
 * atomic. This is a monotonic operation: all writers should be writing the same data, so which
 * one wins is immaterial.
 *
 * Code adapted from Ian Hummel's code from this PR:
 * https://github.com/themodernlife/spark/commit/4359664b1d557d55b0579023df809542386d5b8c
 */
class DirectOutputCommitter extends OutputCommitter {

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
    val conf = context.getJobConf
    if (shouldCreateSuccessFile(conf)) {
      val outputPath = FileOutputFormat.getOutputPath(conf)
      if (outputPath != null) {
        val fileSys = outputPath.getFileSystem(conf)
        val filePath = new Path(outputPath, FileOutputCommitter.SUCCEEDED_FILE_NAME)
        fileSys.create(filePath).close()
      }
    }
  }

  /** By default, we do create the _SUCCESS file, but we allow it to be turned off. */
  private def shouldCreateSuccessFile(conf: JobConf): Boolean = {
    conf.getBoolean("mapreduce.fileoutputcommitter.marksuccessfuljobs", true)
  }
}
