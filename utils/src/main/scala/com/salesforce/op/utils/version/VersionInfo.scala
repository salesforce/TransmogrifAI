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

package com.salesforce.op.utils.version

import java.util.Properties

import com.salesforce.op.utils.json.JsonLike

import scala.collection.JavaConverters._
import scala.util.Try

/**
 * Container with project version and git related information
 *
 * @param version       project version
 * @param buildTime     build time of this jar
 * @param gitRepo       git repo
 * @param gitBranch     git branch
 * @param gitCommitId   git commit id
 * @param gitCommitTime git commit time
 */
case class VersionInfo
(
  version: Option[String],
  buildTime: Option[String],
  gitRepo: Option[String],
  gitBranch: Option[String],
  gitCommitId: Option[String],
  gitCommitTime: Option[String]
) extends JsonLike

object VersionInfo {

  /**
   * Expecting a properties file to be present as a resource and of the following structure:
   *   version=1.2.3
   *   build.time=2018-01-26 14:02:40 -0800
   *   git.repo=git@github.com:foo/bar.git
   *   git.branch=master
   *   git.commit.id=08b2d82ad212e05f15af383340b94be9af973cd5
   *   git.commit.time=Fri Jan 26 11:21:13 2018 -0800
   */
  private lazy val versionPropsMap: Map[String, String] = Try {
    val prop = new Properties()
    prop.load(this.getClass.getClassLoader.getResourceAsStream("op-version.properties"))
    prop.asScala.toMap
  }.getOrElse(Map.empty)

  private lazy val versionInfo = VersionInfo(
    version = versionPropsMap.get("version"),
    buildTime = versionPropsMap.get("build.time"),
    gitRepo = versionPropsMap.get("git.repo"),
    gitBranch = versionPropsMap.get("git.branch"),
    gitCommitId = versionPropsMap.get("git.commit.id"),
    gitCommitTime = versionPropsMap.get("git.commit.time")
  )

  /**
   * Get current VersionInfo
   */
  def apply(): VersionInfo = versionInfo

}
