module Utils
    using HDF5
    using LinearAlgebra 
    using Statistics
    function load_data()
        n = 11
        y1 = zeros(128,128,128,n)
        let id = "0000"
        i = 0
        j = 1
        while i < n
            id = id[1:end-j] * "$i"
            fid = h5open("Data/scalarHIT128flow" * id * ".h5","r")
            y1[:,:,:,i+1] = read(fid["Y1"])
            i = i + 1
            j = floor(Int, log10(i)) + 1
        end
        end
        return y1
    end

    function specialmod(n,r1)
        if n == 0 || n == r1
            return r1
        else
            return mod(n,r1)
        end
    end

    function coarse_grain(y1)
        n = size(y1,4)
        dd = 4
        r = div(128,dd)
        y2 = zeros(r,r,r,n)
        for i=1:r
            for j=1:r
                for k=1:r
                    for l=1:n
                        y2[i,j,k,l] = mean(y1[1+dd*(i-1):dd*i,1+dd*(j-1):dd*j,1+dd*(k-1):dd*k,l])
                    end
                end
            end
        end
        return y2
    end

    function reorg(y2, samples, distance = 3)
        d1 = Any[]
        push!(d1,[0,0,0])
        push!(d1,[1,0,0])
        push!(d1,[0,1,0])
        push!(d1,[0,0,1])
        for i in 1:size(d1)[1]
            push!(d1,-1*d1[i])
        end
        d1 = unique(d1)

        r = size(y2,1)
        iv = 1:distance:(r-distance)
        y = zeros(length(iv)^3*length(samples),size(d1)[1])
        let ii = 1
        for i in iv
            for j in iv
                for k in iv
                    for p in 1:length(samples)
                        for l = 1:size(d1)[1]
                            ind  = specialmod.([i,j,k] + d1[l],r)
                            y[ii,l] = y2[ind...,samples[p]]
                        end
                        ii = ii + 1
                    end
                end
            end
        end
        end
        y = y'
        return y
    end
end
