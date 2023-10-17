import sys
import hashlib
import base64
import os

def install_mitre_certs(cafile,mitre_chain):
    """Appends MITRE certificate chain to Python's certificate authority file."""

    with open(cafile, "r") as f:
        sha256es_cafile = get_hashes(f.read())
    sha256es_cafile_mitre = get_hashes(mitre_chain)

    # assume we've got the MITRE certs installed already
    has_mitre_certs = True

    print("[{}] --- Checking for MITRE certs".format(cafile))
    for sha256 in sha256es_cafile_mitre:
        if sha256 not in sha256es_cafile:
            # if we don't have one of the MITRE certs installed, we're gonna get 'em all!
            has_mitre_certs = False
    
    if not len(sha256es_cafile_mitre):
        #There aren't built in functions to parse and fingerprint these, so skipping for now.
        print ('No hashes in MITRE cert file, just adding them all')
        has_mitre_certs = False

    if has_mitre_certs:
        print("[{}] --- All MITRE certs found.".format(cafile))
    else:
        print(
            '[{}] --- Missing at least one MITRE cert. Appending.'.format(
                cafile
            )
        )
        with open(cafile, "a") as f:
            f.write(mitre_chain)

#More complicated than necessary, but there's no builtin.  Avoiding more libs
def get_hashes(cafile_contents):
    sha256es = []
    lines = cafile_contents.split("\n")
    cert = ''
    for line in lines:
        if line.startswith('#') or line.startswith(' '):
            continue
        elif line.startswith('-----BEGIN'):
            cert = ''
        elif line.startswith('-----END'):
            crt = base64.b64decode(cert)
            sha256es.append(hashlib.sha256(crt).hexdigest())
        else:
            cert += line
    return sha256es

def main():

    with open(sys.argv[1],'r') as f:
        chain = f.read()

    #This is useful for things like boto3/awscli
    #It may make the requests install unnecessary, but try anyways
    try:
        import certifi
        cacert = certifi.where()
        install_mitre_certs(cacert, chain)
        print ('Certifi library updated')
    except ImportError:
        print ('Certifi library is not installed.  If it is installed later, rerun this script.')

    try:
        import requests
        cacert = requests.certs.where()
        install_mitre_certs(cacert,chain)
        print ('Requests library updated.')
    except ImportError:
        print ('Requests library is not installed.  If it is installed later, rerun this script.')

    #pip decided to include its own copy of certifi.
    # at least on some version of python it won't use a system certifi patched above
    try:
        import pip._vendor.certifi
        cacert = pip._vendor.certifi.where()
        install_mitre_certs(cacert,chain)
        print ('pip certificates updated')
    except ImportError:
        print ('pip does not require its own cert on this version')

    if 'REQUESTS_CA_BUNDLE' in os.environ:
        print ('WARNING: REQUESTS_CA_BUNDLE is set.  It may override any certificates that were just updated.')
    if 'CURL_CA_BUNDLE' in os.environ:
        print ('WARNING: CURL_CA_BUNDLE is set.  It may override any certificates that were just updated.')
    if 'PIP_CERT' in os.environ:
        print ('WARNING: PIP_CERT is set.  It may override any certificates that were just updated.')

if __name__ == "__main__":
    main()
