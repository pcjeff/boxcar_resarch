import wget 
import os
import multiprocessing as mp
import glob

def download_url(url, out_dir=None):
    img_filename = url.strip().split('/')[-1]
    dest_path = os.path.join(out_dir, img_filename)
    if not os.path.exists(dest_path):
        print 'download {}'.format(url)
        try:
            os.system('wget {} -P {} -T 2 -t 2'.format(url, out_dir))
            #wget.download(url, dest_path)
        except:
            print 'fail {}'.format(url)
    else:
        print 'omit {}'.format(url)


def crawlImages(url_list_file, out_dir, multi=False):
    with open(url_list_file, 'r') as f:
        urls = map(str.strip, f.readlines())

        if multi:
            pool = mp.Pool(8)
            for url in urls:
                pool.apply_async(download_url, (url, out_dir))
            pool.close()
            pool.join()
        else:
            for url in urls:
                download_url(url, out_dir)

if __name__ == '__main__':
    
    root_dir = '/tmp3/pcjeff/boxcar/'
    img_dir = '/tmp3/pcjeff/boxcar/webimg'
    multi = True
    #url_list_files = glob.glob(os.path.join(root_dir, 'url_list') + '/*.txt')
    url_list_files = ['/tmp3/pcjeff/boxcar/url_list/skodaoctaviasedan.html_imageurls.txt']

    for url_list_file in url_list_files:
        print url_list_file.split('/')[-1].split('.')[0]
        out_dir = os.path.join(img_dir ,url_list_file.split('/')[-1].split('.')[0])
        try:
            os.mkdir(out_dir)
        except:
            pass

        crawlImages(url_list_file, out_dir, multi)

        
