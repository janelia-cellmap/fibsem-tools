#!/usr/bin/env bash
smbclient "//dm11.hhmi.org/public" -c "cd for_mark_k; mget Z0720-07m_BR_Sec22/*.dat"
smbclient "//dm11.hhmi.org/public" -c "cd for_mark_k; mget Z0720-07m_BR_Sec18/*.dat"
