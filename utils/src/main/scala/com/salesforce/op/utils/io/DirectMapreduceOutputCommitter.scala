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
