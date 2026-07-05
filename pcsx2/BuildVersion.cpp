// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0+

#include "svnrev.h"

#include <cstdio>

namespace BuildVersion
{
	const char* GitTag = GIT_TAG;
	bool GitTaggedCommit = GIT_TAGGED_COMMIT;
	int GitTagHi = 2;
	int GitTagMid = 7;
	int GitTagLo = 350;
	int ARMSX2Build = 2;

	namespace
	{
		char s_git_rev_buf[32];
		const char* MakeGitRev()
		{
			std::snprintf(s_git_rev_buf, sizeof(s_git_rev_buf), "%d.%d.%d.%d-SNAPSHOT",
				GitTagHi, GitTagMid, GitTagLo, ARMSX2Build);
			return s_git_rev_buf;
		}
	} // namespace

	// Initialized after GitTagHi/Mid/Lo/ARMSX2Build above (same TU, declaration order),
	// so MakeGitRev() sees their final values.
	const char* GitRev = MakeGitRev();
	const char* GitHash = GIT_HASH;
	const char* GitDate = GIT_DATE;
} // namespace BuildVersion
