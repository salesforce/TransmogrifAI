/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.test

import org.apache.log4j.{AppenderSkeleton, Level, spi}

import scala.collection.mutable

class MemoryAppender extends AppenderSkeleton {
  private val _logs = new mutable.HashSet[spi.LoggingEvent]

  override def requiresLayout: Boolean = true

  /**
   * Clear out the logs in log collection
   * @return Unit
   */
  override def close(): Unit = {
    _logs.clear
  }

  /**
   * Add a log to the log collection
   * @param event The log event
   * @return Unit
   */
  override def append(event: spi.LoggingEvent): Unit = {
    _logs += event
  }

  /**
   * Log event collection
   * @return A collection of log events
   */
  def logs: mutable.HashSet[spi.LoggingEvent] = _logs

  /**
   * Check if a log exists in the log collection
   * @param logLevel The log level of the message
   * @param logMessage The log message
   * @return Boolean of log existence
   */
  def logExists(logLevel: Level, logMessage: String): Boolean = {
    logs.filter(x => x.getLevel == logLevel).map(x => x.getMessage.toString).contains(logMessage)
  }
}
